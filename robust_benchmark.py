import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from condition_aware_fno import (
    ConditionAwareFNO2d,
    FNO2d,
    compute_adaptive_mask,
    evaluate,
    train_epoch,
)
from datasets import DATASET_REGISTRY, get_dataset
from extensions import (
    AnisotropicCAFNO2d,
    DynamicCAFNO2d,
    LearnableLambdaCAFNO2d,
    PerLayerCAFNO2d,
    compute_full_spectrum_mask,
    compute_learnable_lambda,
    geometric_schedule,
    mask_anisotropy_stats,
    train_with_dynamic_updates,
)

ALL_MODELS = ["std", "ca", "ll", "per_layer", "dynamic", "anisotropic"]

def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False                                                                             

def _build_loaders(dataset_name, args, seed):
    train_ds = get_dataset(dataset_name, n_samples=args.n_train,
                           grid_size=args.grid_size, seed=seed)
    test_ds = get_dataset(dataset_name, n_samples=args.n_test,
                          grid_size=args.grid_size, seed=seed + 1000)

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, generator=gen)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    stats_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader, stats_loader

def _collect_tensors(stats_loader, device):
    a_all, u_all = [], []
    for a, u in stats_loader:
        a_all.append(a)
        u_all.append(u)
    a_all = torch.cat(a_all, 0).to(device)
    u_all = torch.cat(u_all, 0).to(device)
    return a_all, u_all

                                                                               
                          
                                                                               

def _train_standard(model, train_loader, test_loader, device,
                    u_mean, u_std, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    train_losses, test_losses = [], []
    t0 = time.time()
    for _ in range(args.epochs):
        tl = train_epoch(model, train_loader, optimizer, criterion,
                         device, u_mean, u_std)
        vl = evaluate(model, test_loader, criterion, device, u_mean, u_std)
        train_losses.append(tl)
        test_losses.append(vl)
    return {
        "final_test_loss": test_losses[-1],
        "train_losses": train_losses,
        "test_losses": test_losses,
        "training_time": time.time() - t0,
    }
def run_trial(dataset_name: str, seed: int, args,
              device: torch.device) -> dict:
    set_seeds(seed)

                                                                              
    train_loader, test_loader, stats_loader = _build_loaders(
        dataset_name, args, seed
    )
    a_all, u_all = _collect_tensors(stats_loader, device)

    u_mean = u_all.mean().item()
    u_std = u_all.std().item()
    u_norm = (u_all - u_mean) / (u_std + 1e-8)

                                 
    n_mask = min(200, len(a_all))
    a_mask = a_all[:n_mask]
    u_mask = u_norm[:n_mask]                          
    low_mask = high_mask = None
    low_lam = high_lam = None
    full_low_mask = full_high_mask = None

    needs_base_mask = bool(
        set(args.models) & {"ca", "ll", "per_layer", "dynamic"}
    )
    needs_ll = "ll" in args.models
    needs_full = "anisotropic" in args.models

    if needs_base_mask:
        low_mask, high_mask = compute_adaptive_mask(
            a_mask, u_mask, modes1=args.modes, modes2=args.modes,
            energy_fraction=args.energy_fraction,
        )

    if needs_ll:
                                                             
        n_half = n_mask // 2
        a_tr, u_tr = a_mask[:n_half], u_mask[:n_half]
        a_val, u_val = a_mask[n_half:], u_mask[n_half:]
        t_ll = time.time()
        low_lam, high_lam, low_mask_ll, high_mask_ll = compute_learnable_lambda(
            a_tr, u_tr, a_val, u_val,
            modes1=args.modes, modes2=args.modes,
            n_iters=args.bilevel_iters,
            energy_fraction=args.energy_fraction,
        )
        ll_precomp_time = time.time() - t_ll
    else:
        low_mask_ll = high_mask_ll = None
        ll_precomp_time = 0.0

    if needs_full:
                                                           
        H_fft = args.grid_size
        W_fft = args.grid_size // 2 + 1
        full_low_mask, full_high_mask = compute_full_spectrum_mask(
            a_mask, u_mask, H_fft=H_fft, W_fft=W_fft,
            energy_fraction=args.energy_fraction,
        )
        aniso_stats = mask_anisotropy_stats(full_low_mask, full_high_mask)
    else:
        aniso_stats = {}

    row = {
        "Dataset": dataset_name,
        "Seed": seed,
        "ll_precomp_time": ll_precomp_time,
        "aniso_aspect_ratio": aniso_stats.get("aspect_ratio", float("nan")),
        "aniso_frac_low": aniso_stats.get("frac_low", float("nan")),
    }

                                                                               
    if "std" in args.models:
        set_seeds(seed)
        model = FNO2d(modes1=args.modes, modes2=args.modes,
                      width=args.width).to(device)
        res = _train_standard(model, train_loader, test_loader,
                              device, u_mean, u_std, args)
        row["std_loss"] = res["final_test_loss"]
        row["std_time"] = res["training_time"]

                                                                               
    if "ca" in args.models:
        set_seeds(seed)
        model = ConditionAwareFNO2d(
            low_mask=low_mask, high_mask=high_mask,
            modes1=args.modes, modes2=args.modes, width=args.width,
        ).to(device)
        res = _train_standard(model, train_loader, test_loader,
                              device, u_mean, u_std, args)
        row["ca_loss"] = res["final_test_loss"]
        row["ca_time"] = res["training_time"]

                                                                            
    if "ll" in args.models:
        set_seeds(seed)
        model = LearnableLambdaCAFNO2d(
            low_mask=low_mask_ll, high_mask=high_mask_ll,
            modes1=args.modes, modes2=args.modes, width=args.width,
        ).to(device)
        res = _train_standard(model, train_loader, test_loader,
                              device, u_mean, u_std, args)
        row["ll_loss"] = res["final_test_loss"]
        row["ll_time"] = res["training_time"] + ll_precomp_time

                                                                            
    if "per_layer" in args.models:
        set_seeds(seed)
        model = PerLayerCAFNO2d(
            low_mask=low_mask, high_mask=high_mask,
            modes1=args.modes, modes2=args.modes, width=args.width,
        ).to(device)

                                                                               
        warm_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        warm_criterion = nn.MSELoss()
        n_warm = max(1, args.epochs // 10)
        for _ in range(n_warm):
            train_epoch(model, train_loader, warm_optimizer, warm_criterion,
                        device, u_mean, u_std)

        print(f"  [PerLayer] Bootstrap masks after {n_warm} warm-start epochs…")
        model.bootstrap_layer_masks(
            train_loader, device,
            energy_fraction=args.energy_fraction,
            n_batches=args.bootstrap_batches,
        )

                                               
        res = _train_standard(model, train_loader, test_loader,
                              device, u_mean, u_std, args)
        row["per_layer_loss"] = res["final_test_loss"]
        row["per_layer_time"] = res["training_time"]

                                                                               
    if "dynamic" in args.models:
        set_seeds(seed)
        model = DynamicCAFNO2d(
            low_mask=low_mask, high_mask=high_mask,
            modes1=args.modes, modes2=args.modes, width=args.width,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        schedule = geometric_schedule(args.epochs, n_updates=args.dynamic_updates)
        res = train_with_dynamic_updates(
            model, train_loader, test_loader, optimizer, criterion,
            device, u_mean, u_std,
            n_epochs=args.epochs,
            update_schedule=schedule,
            energy_fraction=args.energy_fraction,
        )
        row["dynamic_loss"] = res["final_test_loss"]
        row["dynamic_time"] = res["training_time"]
        row["dynamic_n_updates"] = len(res["update_events"])

                                                                               
    if "anisotropic" in args.models:
        set_seeds(seed)
        model = AnisotropicCAFNO2d(
            low_mask=full_low_mask, high_mask=full_high_mask,
            width=args.width,
        ).to(device)
        res = _train_standard(model, train_loader, test_loader,
                              device, u_mean, u_std, args)
        row["anisotropic_loss"] = res["final_test_loss"]
        row["anisotropic_time"] = res["training_time"]

    return row

_LOSS_COLS = {
    "std": "std_loss",
    "ca": "ca_loss",
    "ll": "ll_loss",
    "per_layer": "per_layer_loss",
    "dynamic": "dynamic_loss",
    "anisotropic": "anisotropic_loss",
}

_LABEL = {
    "std": "Standard FNO",
    "ca": "CA-FNO (static)",
    "ll": "CA-FNO + Learned λ",
    "per_layer": "CA-FNO per-layer",
    "dynamic": "CA-FNO dynamic",
    "anisotropic": "CA-FNO anisotropic",
}

def _aggregate(df: pd.DataFrame, models: list) -> pd.DataFrame:
    agg_rows = []
    for dataset in df["Dataset"].unique():
        sub = df[df["Dataset"] == dataset]
        row = {"Dataset": dataset}
        ref_col = _LOSS_COLS.get("std")
        ref_mean = sub[ref_col].mean() if ref_col in sub.columns else None
        for m in models:
            col = _LOSS_COLS[m]
            if col not in sub.columns:
                continue
            vals = sub[col].dropna()
            mean = vals.mean()
            std = vals.std()
            win_vs_std = (
                (sub[col] < sub[_LOSS_COLS["std"]]).mean() * 100
                if "std" in models and _LOSS_COLS["std"] in sub.columns
                else float("nan")
            )
            imp = (
                (ref_mean - mean) / ref_mean * 100
                if ref_mean and ref_mean > 0
                else float("nan")
            )
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_imp%"] = imp
            row[f"{m}_win%"] = win_vs_std
        agg_rows.append(row)
    return pd.DataFrame(agg_rows)

def _plot_convergence(df: pd.DataFrame, output_dir: Path, models: list) -> None:
    datasets = df["Dataset"].unique()
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5), sharey=False)
    if n_ds == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = df[df["Dataset"] == dataset]
        data = []
        labels = []
        for m in models:
            col = _LOSS_COLS[m]
            if col in sub.columns:
                data.append(sub[col].dropna().values)
                labels.append(_LABEL[m])
        ax.boxplot(data, labels=labels, patch_artist=True)
        ax.set_title(dataset.upper())
        ax.set_ylabel("Final Test MSE Loss")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    fig.suptitle("Final test loss distribution across seeds", fontsize=13)
    plt.tight_layout()
    path = output_dir / "benchmark_boxplot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Boxplot saved → {path}")

def _plot_improvement(summary: pd.DataFrame, output_dir: Path,
                      models: list) -> None:
    non_std = [m for m in models if m != "std"]
    if not non_std:
        return

    datasets = summary["Dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.8 / max(len(non_std), 1)

    fig, ax = plt.subplots(figsize=(max(7, 2 * len(datasets)), 5))
    for i, m in enumerate(non_std):
        col = f"{m}_imp%"
        if col not in summary.columns:
            continue
        vals = summary[col].fillna(0).values
        ax.bar(x + i * width, vals, width, label=_LABEL[m], alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x + width * (len(non_std) - 1) / 2)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.set_ylabel("Mean improvement over std FNO (%)")
    ax.set_title("Performance improvement by extension")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = output_dir / "benchmark_improvement.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Improvement chart saved → {path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark all CA-FNO variants across datasets and seeds"
    )
                          
    parser.add_argument("--datasets", nargs="+",
                        default=list(DATASET_REGISTRY.keys()),
                        choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS,
                        help="Which model variants to include")
                         
    parser.add_argument("--trials", type=int, default=20,
                        help="Random seeds per (dataset, model)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument("--n_train", type=int, default=800)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed; trial i uses seed+i")
                                           
    parser.add_argument("--energy_fraction", type=float, default=0.95,
                        help="η for energy criterion (all extensions)")
    parser.add_argument("--bilevel_iters", type=int, default=60,
                        help="Outer iterations for learnable-lambda (Ext 1)")
    parser.add_argument("--bilevel_lr", type=float, default=0.05,
                        help="Outer step size for learnable-lambda (Ext 1)")
    parser.add_argument("--bootstrap_batches", type=int, default=20,
                        help="Mini-batches used for per-layer bootstrap (Ext 2)")
    parser.add_argument("--dynamic_updates", type=int, default=4,
                        help="Number of mask re-evaluations for dynamic (Ext 3)")
               
    parser.add_argument("--output_dir", type=str, default="benchmark_results")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device        : {device}")
    print(f"Datasets      : {args.datasets}")
    print(f"Models        : {args.models}")
    print(f"Trials/dataset: {args.trials}")
    print(f"Epochs        : {args.epochs}")
    print()

    rows = []
    for dataset in args.datasets:
        print("=" * 72)
        print(f"Dataset: {dataset.upper()}")
        print("=" * 72)
        pbar = tqdm(range(args.trials), desc=dataset)
        for i in pbar:
            seed = args.seed + i
            try:
                row = run_trial(dataset, seed, args, device)
                rows.append(row)

                                                                              
                status_parts = []
                for m in args.models:
                    col = _LOSS_COLS[m]
                    if col in row:
                        status_parts.append(f"{m}={row[col]:.4e}")
                pbar.set_description(f"{dataset} seed={seed} | " + "  ".join(status_parts))

            except Exception as exc:
                print(f"\n[WARN] trial failed dataset={dataset} seed={seed}: {exc}")
                import traceback; traceback.print_exc()

    if not rows:
        print("No successful trials. Exiting.")
        return

    df = pd.DataFrame(rows)
    raw_path = output_dir / "benchmark_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results saved → {raw_path}")

    summary = _aggregate(df, args.models)
    summary_path = output_dir / "benchmark_summary.csv"
    summary.round(4).to_csv(summary_path, index=False)

    print("\n" + "=" * 72)
    print("AGGREGATE RESULTS")
    print("=" * 72)
                          
    print_cols = ["Dataset"]
    for m in args.models:
        for suffix in ["_mean", "_std", "_imp%", "_win%"]:
            c = f"{m}{suffix}"
            if c in summary.columns:
                print_cols.append(c)
    print(summary[print_cols].round(4).to_string(index=False))

                                                                               
    if "anisotropic" in args.models and "aniso_aspect_ratio" in df.columns:
        print("\nAnisotropic mask statistics (mean over trials):")
        for ds in df["Dataset"].unique():
            sub = df[df["Dataset"] == ds]
            ar = sub["aniso_aspect_ratio"].dropna().mean()
            fl = sub["aniso_frac_low"].dropna().mean()
            print(f"  {ds:12s}  aspect_ratio={ar:.3f}  frac_low_retained={fl:.3f}")
        print()

                                                                               
    _plot_convergence(df, output_dir, args.models)
    _plot_improvement(summary, output_dir, args.models)

    print(f"\nAll outputs in: {output_dir}/")

if __name__ == "__main__":
    main()