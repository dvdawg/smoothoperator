from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_adaptive_mask(
    a_batch: torch.Tensor,
    u_batch: torch.Tensor,
    modes1: int,
    modes2: int,
    lambda_reg: float = 1e-6,
    energy_fraction: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del a_batch

    device = u_batch.device
    B, H, W = u_batch.shape

    u_fft = torch.fft.rfft2(u_batch)
    _, H_fft, W_fft = u_fft.shape
    m2_eff = min(modes2, W_fft)

    def compute_energy(i_start: int, i_end: int) -> torch.Tensor:
        m1 = max(i_end - i_start, 0)
        energy = torch.zeros((m1, modes2), device=device)
        for i_abs in range(i_start, i_end):
            i_rel = i_abs - i_start
            for j in range(m2_eff):
                Y = u_fft[:, i_abs, j]
                e = torch.sum(torch.abs(Y) ** 2).real
                energy[i_rel, j] = e
        return energy

    def select_mask(scores: torch.Tensor) -> torch.Tensor:
        flat = scores.reshape(-1)
        total = flat.sum()
        if total <= 0:
            return torch.ones_like(scores, dtype=torch.bool, device=device)

        sorted_vals, sorted_idx = torch.sort(flat, descending=True)
        cumsum = torch.cumsum(sorted_vals, dim=0)
        target = energy_fraction * total

        k = int((cumsum <= target).sum().item())
        if k == 0:
            k = 1
        if k > flat.numel():
            k = flat.numel()

        mask_flat = torch.zeros_like(flat, dtype=torch.bool, device=device)
        mask_flat[sorted_idx[:k]] = True
        return mask_flat.view_as(scores)

    low_start = 0
    low_end = min(modes1, H_fft)
    high_end = H_fft
    high_start = max(high_end - modes1, 0)

    low_scores = compute_energy(low_start, low_end)
    high_scores = compute_energy(high_start, high_end)

    low_mask = select_mask(low_scores)
    high_mask = select_mask(high_scores)
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
    device = a_batch.device
    B, H, W = a_batch.shape

    a_fft = torch.fft.rfft2(a_batch)
    u_fft = torch.fft.rfft2(u_batch)
    _, H_fft, W_fft = a_fft.shape
    m2_eff = min(modes2, W_fft)

    def init_region(mask: torch.Tensor, i_start: int, i_end: int) -> torch.Tensor:
        m1 = max(i_end - i_start, 0)
        weights = torch.zeros((m1, modes2), dtype=torch.cfloat, device=device)
        for i_abs in range(i_start, i_end):
            i_rel = i_abs - i_start
            for j in range(m2_eff):
                if i_rel < mask.shape[0] and j < mask.shape[1] and mask[i_rel, j]:
                    X = a_fft[:, i_abs, j]
                    Y = u_fft[:, i_abs, j]
                    XHX = torch.sum(X.conj() * X).real
                    XHY = torch.sum(X.conj() * Y)
                    denom = XHX + lambda_reg
                    w = XHY / (denom + 1e-12)
                    weights[i_rel, j] = w
        return weights

    low_start = 0
    low_end = min(modes1, H_fft)
    high_end = H_fft
    high_start = max(high_end - modes1, 0)

    low_weights = init_region(low_mask, low_start, low_end)
    high_weights = init_region(high_mask, high_start, high_end)
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
        # Add channel dimension if input is (batch, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(-1)  # (batch, H, W) -> (batch, H, W, 1)
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
        x = x.squeeze(-1)  # Remove channel dimension: (batch, H, W, 1) -> (batch, H, W)
        return x


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
        # Add channel dimension if input is (batch, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(-1)  # (batch, H, W) -> (batch, H, W, 1)
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
        x = x.squeeze(-1)  # Remove channel dimension: (batch, H, W, 1) -> (batch, H, W)
        return x


def train_epoch(model, train_loader, optimizer, criterion, device, u_mean, u_std):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    for a, u in train_loader:
        a = a.to(device)
        u = u.to(device)
        
        # Normalize target
        u_normalized = (u - u_mean) / (u_std + 1e-8)
        
        # Forward pass
        optimizer.zero_grad()
        u_pred = model(a)
        
        # Compute loss on normalized outputs
        loss = criterion(u_pred, u_normalized)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * a.size(0)
        n_samples += a.size(0)
    
    return total_loss / n_samples


def evaluate(model, test_loader, criterion, device, u_mean, u_std):
    """Evaluate the model on test data"""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for a, u in test_loader:
            a = a.to(device)
            u = u.to(device)
            
            # Normalize target
            u_normalized = (u - u_mean) / (u_std + 1e-8)
            
            # Forward pass
            u_pred = model(a)
            
            # Compute loss on normalized outputs
            loss = criterion(u_pred, u_normalized)
            
            total_loss += loss.item() * a.size(0)
            n_samples += a.size(0)
    
    return total_loss / n_samples
