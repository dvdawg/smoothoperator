"""Generate the test-error convergence figure (relative L2 vs epoch).

Trains the standard and condition-aware FNO on a few representative datasets,
averaging per-epoch test relative-L2 over a handful of seeds, and saves
writeup/images/convergence.png.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import get_dataset, dataset_in_channels
from condition_aware_fno import (
    FNO2d, ConditionAwareFNO2d, compute_adaptive_mask, train_epoch, evaluate,
)
from torch.utils.data import DataLoader


def curves_for(dataset, seeds, epochs, device, grid=64, n_train=800, n_test=200,
               modes=12, width=64, eta=0.95):
    in_ch = dataset_in_channels(dataset)
    std_all, ca_all = [], []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        tr = get_dataset(dataset, n_samples=n_train, grid_size=grid, seed=seed)
        te = get_dataset(dataset, n_samples=n_test, grid_size=grid, seed=seed + 1000)
        trl = DataLoader(tr, batch_size=20, shuffle=True)
        tel = DataLoader(te, batch_size=20, shuffle=False)
        A = torch.stack([tr[i][0] for i in range(len(tr))])
        U = torch.stack([tr[i][1] for i in range(len(tr))])
        u_mean, u_std = U.mean().item(), U.std().item()
        un = (U - u_mean) / (u_std + 1e-8)
        low, high = compute_adaptive_mask(A[:200].cpu(), un[:200].cpu(), modes, modes, eta)

        def run(model):
            model = model.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            crit = nn.MSELoss()
            curve = []
            for _ in range(epochs):
                train_epoch(model, trl, opt, crit, device, u_mean, u_std)
                curve.append(evaluate(model, tel, crit, device, u_mean, u_std))
            return curve

        torch.manual_seed(seed)
        std_all.append(run(FNO2d(in_channels=in_ch, modes1=modes, modes2=modes, width=width)))
        torch.manual_seed(seed)
        ca_all.append(run(ConditionAwareFNO2d(low, high, in_channels=in_ch,
                                              modes1=modes, modes2=modes, width=width)))
    return np.array(std_all), np.array(ca_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["heat", "wave", "heat_sensor"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", default="../writeup/images/convergence.png")
    args = ap.parse_args()
    device = torch.device("cpu")
    seeds = list(range(42, 42 + args.seeds))

    fig, axes = plt.subplots(1, len(args.datasets), figsize=(5 * len(args.datasets), 4))
    if len(args.datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, args.datasets):
        std, ca = curves_for(ds, seeds, args.epochs, device)
        ep = np.arange(1, args.epochs + 1)
        for arr, lab, col in [(std, "Standard FNO", "C0"), (ca, "CA-FNO", "C1")]:
            m, s = arr.mean(0), arr.std(0)
            ax.plot(ep, m, label=lab, color=col)
            ax.fill_between(ep, m - s, m + s, alpha=0.2, color=col)
        ax.set_title(ds); ax.set_xlabel("epoch"); ax.set_ylabel("test rel. $L_2$")
        ax.set_yscale("log"); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
