import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from tqdm import tqdm

# Import your modules
from datasets import make_pde_dataloaders
from condition_aware_fno import (
    FNO2d, 
    ConditionAwareFNO2d, 
    compute_adaptive_mask, 
    train_epoch, 
    evaluate
)

def run_paired_trial(
    dataset_name: str, 
    seed: int, 
    args
):
    """
    Runs a single paired trial:
    1. Generates data with specific seed.
    2. Trains Baseline FNO.
    3. Trains Condition-Aware FNO.
    Returns test losses for both.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Prepare Data
    train_loader, test_loader = make_pde_dataloaders(
        name=dataset_name,
        n_train=args.n_train,
        n_test=args.n_test,
        grid_size=args.grid_size,
        batch_size=args.batch_size,
        seed=seed
    )
    
    # Pre-calculate stats for normalization
    all_u = []
    all_a = []
    for a, u in train_loader:
        all_u.append(u)
        all_a.append(a)
    
    full_u = torch.cat(all_u).to(device)
    full_a = torch.cat(all_a).to(device)
    
    u_mean, u_std = full_u.mean(), full_u.std()

    # 2. Compute Adaptive Masks (Only needed for CA-FNO, but computed on training data)
    # We use the full training batch for the SVD/Energy calculation
    start_time = time()
    low_mask, high_mask = compute_adaptive_mask(
        full_a, 
        full_u, 
        modes1=args.modes, 
        modes2=args.modes,
        energy_fraction=0.95
    )
    mask_time = time() - start_time

    # --- Train Baseline FNO ---
    model_base = FNO2d(modes1=args.modes, modes2=args.modes, width=args.width).to(device)
    optimizer_base = optim.Adam(model_base.parameters(), lr=args.lr)
    scheduler_base = optim.lr_scheduler.StepLR(optimizer_base, step_size=20, gamma=0.5)
    criterion = nn.MSELoss()

    for ep in range(args.epochs):
        train_epoch(model_base, train_loader, optimizer_base, criterion, device, u_mean, u_std)
        scheduler_base.step()
    
    loss_base = evaluate(model_base, test_loader, criterion, device, u_mean, u_std)

    # --- Train Condition-Aware FNO ---
    model_ca = ConditionAwareFNO2d(
        low_mask=low_mask, 
        high_mask=high_mask, 
        modes1=args.modes, 
        modes2=args.modes, 
        width=args.width
    ).to(device)
    
    optimizer_ca = optim.Adam(model_ca.parameters(), lr=args.lr)
    scheduler_ca = optim.lr_scheduler.StepLR(optimizer_ca, step_size=20, gamma=0.5)

    for ep in range(args.epochs):
        train_epoch(model_ca, train_loader, optimizer_ca, criterion, device, u_mean, u_std)
        scheduler_ca.step()
        
    loss_ca = evaluate(model_ca, test_loader, criterion, device, u_mean, u_std)

    return loss_base, loss_ca, mask_time

def main():
    parser = argparse.ArgumentParser(description="Robust Benchmark for Condition-Aware FNO")
    parser.add_argument("--datasets", nargs="+", default=["poisson", "darcy"], help="List of datasets to test")
    parser.add_argument("--trials", type=int, default=10, help="Number of random seeds to average over")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per trial")
    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument("--n_train", type=int, default=800)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    results = []

    print(f"Starting Benchmark with {args.trials} trials per dataset...")
    print(f"Datasets: {args.datasets}")
    
    # Iterate over datasets
    for dataset in args.datasets:
        print(f"\nEvaluating: {dataset.upper()}")
        
        trial_results = []
        base_losses = []
        ca_losses = []
        
        # Progress bar for trials
        pbar = tqdm(range(args.trials))
        for i in pbar:
            seed = 42 + i # Deterministic sequence of seeds
            
            try:
                l_base, l_ca, m_time = run_paired_trial(dataset, seed, args)
                
                # Calculate improvement percentage, handling zero baseline loss
                # Use a small threshold to avoid numerical issues with near-zero losses
                if l_base < 1e-10:  # Avoid division by zero or near-zero
                    if l_ca < 1e-10:
                        improvement = 0.0  # Both are essentially zero (perfect predictions)
                    else:
                        # Baseline is perfect but CA-FNO is not - improvement undefined
                        # Use NaN to indicate undefined case (pandas will handle this in stats)
                        improvement = float('nan')
                else:
                    improvement = (l_base - l_ca) / l_base * 100
                
                results.append({
                    "Dataset": dataset,
                    "Seed": seed,
                    "Baseline Loss": l_base,
                    "CA-FNO Loss": l_ca,
                    "Improvement (%)": improvement,
                    "Mask Time (s)": m_time
                })
                
                base_losses.append(l_base)
                ca_losses.append(l_ca)
                
                # Format improvement for display (handle NaN)
                imp_str = f"{improvement:+.2f}%" if not np.isnan(improvement) else "N/A"
                pbar.set_description(f"Base: {l_base:.4f} | CA: {l_ca:.4f} | Imp: {imp_str}")
            except Exception as e:
                print(f"Trial {i} failed: {e}")

    # --- Analysis & Visualization ---
    df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("FINAL AGGREGATE RESULTS")
    print("="*50)
    
    # Group by dataset and compute statistics
    summary = df.groupby("Dataset").agg({
        "Baseline Loss": ["mean", "std"],
        "CA-FNO Loss": ["mean", "std"],
        "Improvement (%)": ["mean", "std", lambda x: (x > 0).mean() * 100]
    }).round(4)
    
    # Rename columns for clarity
    summary.columns = [
        "Base Mean", "Base Std", 
        "CA Mean", "CA Std", 
        "Imp Mean (%)", "Imp Std", "Win Rate (%)"
    ]
    
    print(summary)
    
    # Save results
    df.to_csv("benchmark_raw_results.csv", index=False)
    print("\nRaw results saved to 'benchmark_raw_results.csv'")

    # Create comparison plot
    plt.figure(figsize=(10, 6))
    
    # Melt dataframe for Seaborn boxplot
    df_melt = df.melt(id_vars=["Dataset", "Seed"], 
                      value_vars=["Baseline Loss", "CA-FNO Loss"], 
                      var_name="Model", value_name="MSE Loss")
    
    sns.boxplot(x="Dataset", y="MSE Loss", hue="Model", data=df_melt)
    plt.title(f"Model Comparison over {args.trials} seeds")
    plt.grid(True, alpha=0.3)
    plt.savefig("benchmark_boxplot.png")
    print("Boxplot saved to 'benchmark_boxplot.png'")

if __name__ == "__main__":
    main()