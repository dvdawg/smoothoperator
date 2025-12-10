import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from condition_aware_fno import (
    ConditionAwareFNO2d,
    FNO2d,
    compute_adaptive_mask,
    evaluate,
    ridge_regression_init,
    train_epoch,
)
from datasets import DATASET_REGISTRY, get_dataset


def set_seeds(seed: int):
    """Deterministic seeding helper."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_and_eval(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    u_mean: float,
    u_std: float,
    lr: float,
    epochs: int,
):
    """Train a model and return losses plus timing."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, test_losses = [], []
    start = time.time()
    for _ in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, u_mean, u_std)
        test_loss = evaluate(model, test_loader, criterion, device, u_mean, u_std)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    duration = time.time() - start
    return {
        "final_test_loss": test_losses[-1],
        "train_losses": train_losses,
        "test_losses": test_losses,
        "training_time": duration,
    }


def run_trial(dataset_name: str, seed: int, args, device: torch.device):
    """Single seed trial for one dataset."""
    set_seeds(seed)

    # Build datasets
    train_dataset = get_dataset(
        dataset_name, n_samples=args.n_train, grid_size=args.grid_size, seed=seed
    )
    test_dataset = get_dataset(
        dataset_name, n_samples=args.n_test, grid_size=args.grid_size, seed=seed + 1000
    )

    # Deterministic data loaders
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Stats and mask computation use non-shuffled data for reproducibility
    stats_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    a_all, u_all = [], []
    for a, u in stats_loader:
        a_all.append(a)
        u_all.append(u)
    a_all = torch.cat(a_all, dim=0).to(device)
    u_all = torch.cat(u_all, dim=0).to(device)

    u_mean = u_all.mean().item()
    u_std = u_all.std().item()
    u_all_norm = (u_all - u_mean) / (u_std + 1e-8)

    n_mask = min(200, len(a_all))
    a_mask = a_all[:n_mask]
    u_mask = u_all_norm[:n_mask]

    mask_start = time.time()
    low_mask, high_mask = compute_adaptive_mask(
        a_mask, u_mask, modes1=args.modes, modes2=args.modes, energy_fraction=args.energy_fraction
    )
    mask_time = time.time() - mask_start

    # Ridge regression init to match test_suite.py
    init_low, init_high = ridge_regression_init(
        a_mask,
        u_mask,
        low_mask,
        high_mask,
        modes1=args.modes,
        modes2=args.modes,
        in_channels=args.width,
        out_channels=args.width,
        lambda_reg=args.lambda_reg,
    )

    # Standard FNO
    model_std = FNO2d(modes1=args.modes, modes2=args.modes, width=args.width).to(device)
    std_results = train_and_eval(
        model_std, train_loader, test_loader, device, u_mean, u_std, args.lr, args.epochs
    )

    # Condition-Aware FNO (with masks + ridge init)
    model_ca = ConditionAwareFNO2d(
        low_mask=low_mask,
        high_mask=high_mask,
        low_weights=init_low,
        high_weights=init_high,
        modes1=args.modes,
        modes2=args.modes,
        width=args.width,
    ).to(device)
    ca_results = train_and_eval(
        model_ca, train_loader, test_loader, device, u_mean, u_std, args.lr, args.epochs
    )

    base_loss = std_results["final_test_loss"]
    ca_loss = ca_results["final_test_loss"]
    improvement = (base_loss - ca_loss) / base_loss * 100 if base_loss > 0 else 0.0

    return {
        "Dataset": dataset_name,
        "Seed": seed,
        "Baseline Loss": base_loss,
        "CA-FNO Loss": ca_loss,
        "Improvement (%)": improvement,
        "Mask Time (s)": mask_time,
        "Std Train Time": std_results["training_time"],
        "CA Train Time": ca_results["training_time"],
    }


def main():
    parser = argparse.ArgumentParser(description="Robust benchmark for Condition-Aware FNO")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_REGISTRY.keys()),
        choices=list(DATASET_REGISTRY.keys()),
        help="Datasets to evaluate",
    )
    parser.add_argument("--trials", type=int, default=20, help="Number of seeds per dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per model")
    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument("--n_train", type=int, default=800)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--energy_fraction", type=float, default=0.95)
    parser.add_argument("--lambda_reg", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="benchmark_results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    print(f"Datasets: {args.datasets}")
    print(f"Trials per dataset: {args.trials}")

    rows = []
    for dataset in args.datasets:
        print("\n" + "=" * 80)
        print(f"Evaluating dataset: {dataset.upper()}")
        pbar = tqdm(range(args.trials))
        for i in pbar:
            seed = args.seed + i
            try:
                row = run_trial(dataset, seed, args, device)
                rows.append(row)
                pbar.set_description(
                    f"Seed {seed} | Base {row['Baseline Loss']:.4e} | "
                    f"CA {row['CA-FNO Loss']:.4e} | Imp {row['Improvement (%)']:+.2f}%"
                )
            except Exception as exc:
                pbar.set_description(f"Seed {seed} failed: {exc}")
                print(f"[WARN] Trial failed for dataset={dataset}, seed={seed}: {exc}")

    if not rows:
        print("No successful trials recorded. Exiting.")
        return

    df = pd.DataFrame(rows)
    csv_path = output_dir / "benchmark_raw_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nRaw results saved to {csv_path}")

    # Aggregate summary
    summary = df.groupby("Dataset").agg(
        {
            "Baseline Loss": ["mean", "std"],
            "CA-FNO Loss": ["mean", "std"],
            "Improvement (%)": ["mean", "std", lambda x: (x > 0).mean() * 100],
        }
    )
    summary.columns = [
        "Base Mean",
        "Base Std",
        "CA Mean",
        "CA Std",
        "Imp Mean (%)",
        "Imp Std",
        "Win Rate (%)",
    ]
    summary_path = output_dir / "benchmark_summary.csv"
    summary.round(4).to_csv(summary_path)

    print("\n" + "=" * 80)
    print("FINAL AGGREGATE RESULTS")
    print("=" * 80)
    print(summary.round(4))

    # Boxplot comparison
    plt.figure(figsize=(10, 6))
    df_melt = df.melt(
        id_vars=["Dataset", "Seed"],
        value_vars=["Baseline Loss", "CA-FNO Loss"],
        var_name="Model",
        value_name="MSE Loss",
    )
    sns.boxplot(x="Dataset", y="MSE Loss", hue="Model", data=df_melt)
    plt.title(f"Model comparison over {args.trials} seeds")
    plt.grid(True, alpha=0.3)
    boxplot_path = output_dir / "benchmark_boxplot.png"
    plt.savefig(boxplot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Boxplot saved to {boxplot_path}")


if __name__ == "__main__":
    main()