"""
Condition-Aware Fourier Neural Operator (FNO) with Adaptive Spectral Truncation.

This module defines:
- Standard FNO2d
- ConditionAwareFNO2d (with mask-based spectral truncation)
- Utilities for computing adaptive spectral masks and ridge-regression initialization
- A main() entrypoint that trains on the Poisson dataset.
"""

import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets import PoissonDataset


def compute_adaptive_mask(
    a_batch: torch.Tensor,
    u_batch: torch.Tensor,
    modes1: int,
    modes2: int,
    lambda_reg: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute adaptive low- and high-frequency masks based on simple utility scores.

    Currently returns masks that keep all modes in both regions, but per-mode
    ridge-style scores are computed so that stricter truncation can be plugged in.
    """
    device = a_batch.device
    B, H, W = a_batch.shape

    a_fft = torch.fft.rfft2(a_batch)  # [B, H, W_fft]
    u_fft = torch.fft.rfft2(u_batch)  # [B, H, W_fft]
    _, H_fft, W_fft = a_fft.shape

    def compute_scores(i_start: int, i_end: int) -> torch.Tensor:
        scores = torch.zeros((i_end - i_start, modes2), device=device)
        for i_abs in range(i_start, i_end):
            i_rel = i_abs - i_start
            for j in range(modes2):
                X = a_fft[:, i_abs, j]
                Y = u_fft[:, i_abs, j]
                XHX = torch.sum(X.conj() * X).real
                XHY = torch.sum(X.conj() * Y)
                denom = XHX + lambda_reg
                w = XHY / (denom + 1e-12)
                scores[i_rel, j] = torch.abs(w)
        return scores

    low_scores = compute_scores(0, min(modes1, H_fft))
    high_scores = compute_scores(max(H_fft - modes1, 0), H_fft)

    low_mask = torch.ones_like(low_scores, dtype=torch.bool, device=device)
    high_mask = torch.ones_like(high_scores, dtype=torch.bool, device=device)
    return low_mask, high_mask


def ridge_regression_init(
    a_batch: torch.Tensor,
    u_batch: torch.Tensor,
    low_mask: torch.Tensor,
    high_mask: torch.Tensor,
    modes1: int,
    modes2: int,
    lambda_reg: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize spectral weights using complex ridge regression in Fourier space.

    Returns:
        weights1: complex tensor [modes1, modes2] for low frequencies
        weights2: complex tensor [modes1, modes2] for high frequencies
    """
    device = a_batch.device
    B, H, W = a_batch.shape

    a_fft = torch.fft.rfft2(a_batch)
    u_fft = torch.fft.rfft2(u_batch)
    _, H_fft, W_fft = a_fft.shape

    def init_region(mask: torch.Tensor, i_start: int, i_end: int) -> torch.Tensor:
        m1 = i_end - i_start
        weights = torch.zeros((m1, modes2), dtype=torch.cfloat, device=device)
        for i_abs in range(i_start, i_end):
            i_rel = i_abs - i_start
            for j in range(modes2):
                if i_rel < mask.shape[0] and j < mask.shape[1] and mask[i_rel, j]:
                    X = a_fft[:, i_abs, j]
                    Y = u_fft[:, i_abs, j]
                    XHX = torch.sum(X.conj() * X).real
                    XHY = torch.sum(X.conj() * Y)
                    denom = XHX + lambda_reg
                    w = XHY / (denom + 1e-12)
                    weights[i_rel, j] = w
        return weights

    low_weights = init_region(low_mask, 0, min(modes1, H_fft))
    high_weights = init_region(high_mask, max(H_fft - modes1, 0), H_fft)
    return low_weights, high_weights


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        B, C, H_fft, W_fft = x_ft.shape

        out_ft = torch.zeros(
            B, self.out_channels, H_fft, W_fft, dtype=torch.cfloat, device=x.device
        )

        m1 = min(self.modes1, H_fft)
        m2 = min(self.modes2, W_fft)

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )

        x = torch.fft.irfft2(out_ft, s=(H, W))
        return x


class ConditionAwareSpectralConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        low_mask: torch.Tensor,
        high_mask: torch.Tensor,
        modes1: int,
        modes2: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.register_buffer("low_mask", low_mask.bool())
        self.register_buffer("high_mask", high_mask.bool())

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        B, C, H_fft, W_fft = x_ft.shape

        out_ft = torch.zeros(
            B, self.out_channels, H_fft, W_fft, dtype=torch.cfloat, device=x.device
        )

        m1 = min(self.modes1, H_fft)
        m2 = min(self.modes2, W_fft)

        low_mask = self.low_mask[:m1, :m2].to(torch.cfloat)
        high_mask = self.high_mask[:m1, :m2].to(torch.cfloat)

        w1 = self.weights1[:, :, :m1, :m2] * low_mask.unsqueeze(0).unsqueeze(0)
        w2 = self.weights2[:, :, :m1, :m2] * high_mask.unsqueeze(0).unsqueeze(0)

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], w1)
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], w2)

        x = torch.fft.irfft2(out_ft, s=(H, W))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1: int = 12, modes2: int = 12, width: int = 64):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = nn.Linear(1, width)

        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, 1]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x  # [B, H, W, 1]


class ConditionAwareFNO2d(nn.Module):
    def __init__(
        self,
        low_mask: torch.Tensor,
        high_mask: torch.Tensor,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = nn.Linear(1, width)

        self.conv0 = ConditionAwareSpectralConv2d(
            width, width, low_mask, high_mask, modes1, modes2
        )
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, 1]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x  # [B, H, W, 1]


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    u_mean: float,
    u_std: float,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    mean = torch.tensor(u_mean, device=device)
    std = torch.tensor(u_std, device=device)

    for a, u in loader:
        a = a.to(device)
        u = u.to(device)

        u_norm = (u - mean) / (std + 1e-8)

        optimizer.zero_grad()
        pred = model(a.unsqueeze(-1)).squeeze(-1)
        loss = criterion(pred, u_norm)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    u_mean: float,
    u_std: float,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    mean = torch.tensor(u_mean, device=device)
    std = torch.tensor(u_std, device=device)

    with torch.no_grad():
        for a, u in loader:
            a = a.to(device)
            u = u.to(device)

            u_norm = (u - mean) / (std + 1e-8)
            pred = model(a.unsqueeze(-1)).squeeze(-1)
            loss = criterion(pred, u_norm)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    grid_size = 64
    modes = 12
    batch_size = 20
    n_train = 800
    n_test = 200
    n_epochs = 50
    learning_rate = 0.001

    print("Generating synthetic data...")
    train_dataset = PoissonDataset(n_samples=n_train, grid_size=grid_size, seed=42)
    test_dataset = PoissonDataset(n_samples=n_test, grid_size=grid_size, seed=43)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    a_all = []
    u_all = []
    for a, u in train_loader:
        a_all.append(a)
        u_all.append(u)
    for a, u in test_loader:
        a_all.append(a)
        u_all.append(u)
    a_all = torch.cat(a_all, dim=0).to(device)
    u_all = torch.cat(u_all, dim=0).to(device)

    print("Computing normalization statistics and adaptive spectral mask...")
    u_mean = u_all.mean().item()
    u_std = u_all.std().item()
    print(f"Output statistics: mean={u_mean:.4f}, std={u_std:.4f}")
    print("Normalizing outputs to unit scale for training stability...")

    u_all_norm = (u_all - u_mean) / (u_std + 1e-8)

    n_mask = min(200, a_all.shape[0])
    a_mask = a_all[:n_mask]
    u_mask = u_all_norm[:n_mask]

    low_mask, high_mask = compute_adaptive_mask(a_mask, u_mask, modes1=modes, modes2=modes)
    n_active_low = int(low_mask.sum().item())
    n_active_high = int(high_mask.sum().item())
    print(
        f"Adaptive masks: Low={n_active_low}/{modes*modes}, "
        f"High={n_active_high}/{modes*modes} modes selected"
    )

    print("Computing Ridge Regression initialization...")
    init_w1, init_w2 = ridge_regression_init(
        a_mask, u_mask, low_mask, high_mask, modes1=modes, modes2=modes, lambda_reg=1e-4
    )
    print(f"Ridge regression weights computed (shapes: {init_w1.shape}, {init_w2.shape})")

    print("\n" + "=" * 60)
    print("Training Standard FNO (Model A)")
    print("=" * 60)
    model_standard = FNO2d(modes1=modes, modes2=modes, width=64).to(device)
    optimizer_standard = torch.optim.Adam(model_standard.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_losses_standard = []
    test_losses_standard = []

    start_time = time.time()
    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(
            model_standard, train_loader, optimizer_standard, criterion, device, u_mean, u_std
        )
        test_loss = evaluate(
            model_standard, test_loader, criterion, device, u_mean, u_std
        )

        train_losses_standard.append(train_loss)
        test_losses_standard.append(test_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{n_epochs} - "
                f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}"
            )

    standard_time = time.time() - start_time
    print(f"Standard FNO training completed in {standard_time:.2f} seconds")
    print(f"Final Test Loss: {test_losses_standard[-1]:.6f}")

    print("\n" + "=" * 60)
    print("Training Condition-Aware FNO (Model B)")
    print("=" * 60)
    model_adaptive = ConditionAwareFNO2d(
        low_mask=low_mask, high_mask=high_mask, modes1=modes, modes2=modes, width=64
    ).to(device)

    with torch.no_grad():
        if low_mask.any():
            scale = 1.0 / np.sqrt(
                model_adaptive.conv0.in_channels * model_adaptive.conv0.out_channels
            )
            m1, m2 = init_w1.shape
            model_adaptive.conv0.weights1.data[:, :, :m1, :m2] = (
                init_w1 * scale
            ).unsqueeze(0).unsqueeze(0)
        if high_mask.any():
            scale = 1.0 / np.sqrt(
                model_adaptive.conv0.in_channels * model_adaptive.conv0.out_channels
            )
            m1, m2 = init_w2.shape
            model_adaptive.conv0.weights2.data[:, :, :m1, :m2] = (
                init_w2 * scale
            ).unsqueeze(0).unsqueeze(0)

    print("First spectral layer initialized with Ridge Regression weights")

    optimizer_adaptive = torch.optim.Adam(model_adaptive.parameters(), lr=learning_rate)

    train_losses_adaptive = []
    test_losses_adaptive = []

    start_time = time.time()
    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(
            model_adaptive, train_loader, optimizer_adaptive, criterion, device, u_mean, u_std
        )
        test_loss = evaluate(
            model_adaptive, test_loader, criterion, device, u_mean, u_std
        )

        train_losses_adaptive.append(train_loss)
        test_losses_adaptive.append(test_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{n_epochs} - "
                f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}"
            )

    adaptive_time = time.time() - start_time
    print(f"Condition-Aware FNO training completed in {adaptive_time:.2f} seconds")
    print(f"Final Test Loss: {test_losses_adaptive[-1]:.6f}")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print("Standard FNO:")
    print(f"  Final Test Loss: {test_losses_standard[-1]:.6f}")
    print(f"  Training Time: {standard_time:.2f}s\n")

    print("Condition-Aware FNO:")
    print(f"  Final Test Loss: {test_losses_adaptive[-1]:.6f}")
    print(f"  Training Time: {adaptive_time:.2f}s")
    improvement = 100.0 * (test_losses_standard[-1] - test_losses_adaptive[-1]) / max(
        test_losses_standard[-1], 1e-12
    )
    print(f"  Improvement: {improvement:+.2f}%")

    epochs = np.arange(1, n_epochs + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, test_losses_standard, label="Standard FNO")
    plt.plot(epochs, test_losses_adaptive, label="Condition-Aware FNO")
    plt.xlabel("Epoch")
    plt.ylabel("Test MSE Loss (normalized)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fno_comparison.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to 'fno_comparison.png'")


if __name__ == "__main__":
    main()
