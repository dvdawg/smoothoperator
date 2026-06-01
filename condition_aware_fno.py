"""Condition-aware Fourier Neural Operator.

Core model and the data-driven spectral-truncation pre-analysis.

Tensor conventions
------------------
* Inputs are channels-last: ``a`` has shape ``(B, H, W, C_in)`` (``C_in >= 1``).
* Targets ``u`` have shape ``(B, H, W)`` (single output channel).
* Per-mode data matrices stack samples in the rows; for a mode ``k`` the input
  matrix ``X_k`` has shape ``(N, C_in)``.  When ``C_in > 1`` the condition
  number ``kappa_k = sigma_max / sigma_min`` is non-trivial and the
  condition-aware score genuinely differs from a pure energy ranking.  When
  ``C_in = 1`` we have ``kappa_k == 1`` for every mode and the method reduces to
  energy-based adaptive truncation -- a fact we state explicitly rather than
  obscure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _to_channels_first(x: torch.Tensor) -> torch.Tensor:
    """Map a field tensor to (N, C, H, W).

    Accepts (N, H, W) [single channel], (N, H, W, C) [channels-last], or an
    already channels-first (N, C, H, W) tensor (heuristic: a small trailing-vs
    -leading channel axis).  We standardise on channels-last input from the
    datasets, so the (N, H, W, C) branch is the common one.
    """
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4:
        # channels-last (N, H, W, C): C is the small last axis
        return x.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"Unexpected field tensor with shape {tuple(x.shape)}")


def _energy_cond(X_k: torch.Tensor, Y_k: torch.Tensor) -> Tuple[float, float]:
    """Output energy ||Y_k||_F^2 and input condition number kappa(X_k).

    The SVD is computed on CPU: complex ``linalg`` is unsupported on some
    accelerator backends (e.g. Apple MPS), and this scoring is a cheap
    pre-computation regardless of the training device.
    """
    energy = torch.sum(torch.abs(Y_k) ** 2).real.item()
    try:
        s = torch.linalg.svdvals(X_k.detach().cpu())
        if s.numel() > 0 and s[-1].item() > 1e-9:
            cond = (s[0] / s[-1]).item()
        elif s.numel() > 0 and s[0].item() > 1e-12:
            cond = 1e9  # rank-deficient but non-zero: ill-conditioned
        else:
            cond = 1.0  # no signal energy at all
    except Exception:
        cond = 1e9
    return energy, cond


def _select_by_energy_budget(
    scores: torch.Tensor,
    energies: torch.Tensor,
    energy_fraction: float,
) -> torch.Tensor:
    """Smallest high-score set whose cumulative energy reaches the budget.

    Modes are ranked by the utility score (energy / conditioning); we then add
    them in score order until the cumulative *energy* of the selected set
    reaches ``energy_fraction`` of the total.  This realises the paper's rule
    exactly: rank by conditioning-aware utility, stop on an energy budget.  The
    threshold uses ``searchsorted`` so the selected set is the smallest one whose
    cumulative energy is >= the target (no off-by-one undershoot).
    """
    flat_scores = scores.reshape(-1)
    flat_energies = energies.reshape(-1)
    total = flat_energies.sum()
    if total <= 0:
        return torch.ones_like(scores, dtype=torch.bool)

    order = torch.argsort(flat_scores, descending=True)
    cumsum = torch.cumsum(flat_energies[order], dim=0)
    target = energy_fraction * total
    # smallest k with cumsum[k-1] >= target  ->  count of entries strictly below
    k = int((cumsum < target).sum().item()) + 1
    k = max(1, min(k, flat_scores.numel()))

    mask_flat = torch.zeros_like(flat_scores, dtype=torch.bool)
    mask_flat[order[:k]] = True
    return mask_flat.view_as(scores)


# --------------------------------------------------------------------------- #
# Pre-analysis: adaptive mode selection                                       #
# --------------------------------------------------------------------------- #
def compute_adaptive_mask(
    a_batch: torch.Tensor,
    u_batch: torch.Tensor,
    modes1: int,
    modes2: int,
    energy_fraction: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Condition-aware mode selection for the low- and high-wavenumber blocks.

    Returns boolean masks ``(low_mask, high_mask)`` of shape ``(modes1, modes2)``
    selecting the retained modes in the two ``rfft2`` row-blocks used by the
    spectral convolution.
    """
    device = a_batch.device
    a_cf = _to_channels_first(a_batch)
    u_cf = _to_channels_first(u_batch)

    a_fft = torch.fft.rfft2(a_cf)
    u_fft = torch.fft.rfft2(u_cf)
    _, _, H_fft, W_fft = a_fft.shape
    m2_eff = min(modes2, W_fft)

    def _metrics(i_start: int, i_end: int):
        m1 = max(i_end - i_start, 0)
        scores = torch.zeros((m1, modes2), device=device)
        energies = torch.zeros((m1, modes2), device=device)
        for i_abs in range(i_start, i_end):
            i_rel = i_abs - i_start
            for j in range(m2_eff):
                X_k = a_fft[:, :, i_abs, j]  # (N, C_in)
                Y_k = u_fft[:, :, i_abs, j]  # (N, C_out)
                energy, cond = _energy_cond(X_k, Y_k)
                energies[i_rel, j] = energy
                scores[i_rel, j] = energy / (cond + 1e-8)
        return scores, energies

    low_start, low_end = 0, min(modes1, H_fft)
    high_end = H_fft
    high_start = max(high_end - modes1, 0)

    low_s, low_e = _metrics(low_start, low_end)
    high_s, high_e = _metrics(high_start, high_end)

    low_mask = _select_by_energy_budget(low_s, low_e, energy_fraction)
    high_mask = _select_by_energy_budget(high_s, high_e, energy_fraction)
    return low_mask, high_mask


def ridge_diagnostic(
    a_batch: torch.Tensor,
    u_batch: torch.Tensor,
    modes1: int,
    modes2: int,
    lambda_reg: float = 1e-4,
) -> dict:
    """Per-mode ridge regression diagnostic (the linear surrogate of Sec. 3).

    Fits the diagonal-in-Fourier linear surrogate ``u_hat(k) ~ C_k v_hat(k)`` by
    Tikhonov-regularised least squares and returns, per low-block mode, the
    residual fraction and condition number.  This is a *diagnostic / scoring*
    quantity -- it is not the FNO's lifted spectral weight.  Used to validate
    the conditioning analysis (Sec. 7) and to motivate mode selection.
    """
    a_fft = torch.fft.rfft2(_to_channels_first(a_batch))
    u_fft = torch.fft.rfft2(_to_channels_first(u_batch))
    _, C_in, H_fft, W_fft = a_fft.shape
    m1, m2 = min(modes1, H_fft), min(modes2, W_fft)
    out = {"cond": torch.zeros(m1, m2), "resid_frac": torch.zeros(m1, m2)}
    for i in range(m1):
        for j in range(m2):
            X_k = a_fft[:, :, i, j]                 # (N, C_in)
            Y_k = u_fft[:, :, i, j]                 # (N, C_out)
            _, cond = _energy_cond(X_k, Y_k)
            A = X_k.conj().T @ X_k + lambda_reg * torch.eye(C_in, dtype=X_k.dtype, device=X_k.device)
            C_H = torch.linalg.solve(A, X_k.conj().T @ Y_k)
            resid = (X_k @ C_H - Y_k)
            denom = torch.sum(torch.abs(Y_k) ** 2).real + 1e-30
            out["cond"][i, j] = cond
            out["resid_frac"][i, j] = (torch.sum(torch.abs(resid) ** 2).real / denom).item()
    return out


# --------------------------------------------------------------------------- #
# Spectral convolution layers                                                 #
# --------------------------------------------------------------------------- #
class SpectralConv2d(nn.Module):
    """Standard FNO spectral convolution with a fixed rectangular mode box."""

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

    def compl_mul2d(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        _, _, H_fft, W_fft = x_ft.shape
        out_ft = torch.zeros(B, self.out_channels, H_fft, W_fft, dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, H_fft)
        m2 = min(self.modes2, W_fft)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(H, W))


class ConditionAwareSpectralConv2d(nn.Module):
    """Spectral convolution restricted to a condition-aware mode set.

    Optionally carries per-mode Tikhonov coefficients (``lambda_low/high``); the
    ``reg_term`` method returns the corresponding spectral weight penalty so
    that the regularised optimisation analysed in Sec. 7 is actually exercised
    during training (used by the learnable-lambda variant).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        low_mask: torch.Tensor,
        high_mask: torch.Tensor,
        modes1: int,
        modes2: int,
        lambda_low: Optional[torch.Tensor] = None,
        lambda_high: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.register_buffer("low_mask", low_mask.bool())
        self.register_buffer("high_mask", high_mask.bool())

        zeros = torch.zeros(modes1, modes2)
        self.register_buffer("lambda_low", zeros.clone() if lambda_low is None else lambda_low.float())
        self.register_buffer("lambda_high", zeros.clone() if lambda_high is None else lambda_high.float())

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        _, _, H_fft, W_fft = x_ft.shape
        out_ft = torch.zeros(B, self.out_channels, H_fft, W_fft, dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, H_fft)
        m2 = min(self.modes2, W_fft)

        low = self.low_mask[:m1, :m2].to(torch.cfloat)
        high = self.high_mask[:m1, :m2].to(torch.cfloat)
        w1 = self.weights1[:, :, :m1, :m2] * low.unsqueeze(0).unsqueeze(0)
        w2 = self.weights2[:, :, :m1, :m2] * high.unsqueeze(0).unsqueeze(0)

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], w1)
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], w2)
        return torch.fft.irfft2(out_ft, s=(H, W))

    def reg_term(self) -> torch.Tensor:
        """sum_k lambda_k * ||W(k)||_F^2 over retained modes (per-mode Tikhonov)."""
        w1e = (self.weights1.abs() ** 2).sum(dim=(0, 1))  # (modes1, modes2)
        w2e = (self.weights2.abs() ** 2).sum(dim=(0, 1))
        return (self.lambda_low * self.low_mask.float() * w1e).sum() + (
            self.lambda_high * self.high_mask.float() * w2e
        ).sum()


# --------------------------------------------------------------------------- #
# Models                                                                      #
# --------------------------------------------------------------------------- #
class FNO2d(nn.Module):
    """Standard FNO with a fixed rectangular mode truncation (baseline)."""

    def __init__(self, in_channels: int = 1, modes1: int = 12, modes2: int = 12, width: int = 64):
        super().__init__()
        self.modes1, self.modes2, self.width = modes1, modes2, width
        self.fc0 = nn.Linear(in_channels, width)
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
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        x = self.fc0(x).permute(0, 3, 1, 2)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class ConditionAwareFNO2d(nn.Module):
    """CA-FNO: the condition-aware mode mask is applied at *every* spectral layer."""

    def __init__(
        self,
        low_mask: torch.Tensor,
        high_mask: torch.Tensor,
        in_channels: int = 1,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
        lambda_low: Optional[torch.Tensor] = None,
        lambda_high: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.modes1, self.modes2, self.width = modes1, modes2, width
        self.fc0 = nn.Linear(in_channels, width)

        def _mk():
            return ConditionAwareSpectralConv2d(
                width, width, low_mask, high_mask, modes1, modes2, lambda_low, lambda_high
            )

        self.conv0, self.conv1, self.conv2, self.conv3 = _mk(), _mk(), _mk(), _mk()
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        x = self.fc0(x).permute(0, 3, 1, 2)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

    def spectral_regularization(self) -> torch.Tensor:
        total = self.conv0.reg_term()
        for c in (self.conv1, self.conv2, self.conv3):
            total = total + c.reg_term()
        return total


# --------------------------------------------------------------------------- #
# Training / evaluation                                                        #
# --------------------------------------------------------------------------- #
def train_epoch(model, train_loader, optimizer, criterion, device, u_mean, u_std):
    model.train()
    total_loss, n = 0.0, 0
    use_reg = hasattr(model, "spectral_regularization")
    for a, u in train_loader:
        a, u = a.to(device), u.to(device)
        u_norm = (u - u_mean) / (u_std + 1e-8)
        optimizer.zero_grad()
        pred = model(a)
        loss = criterion(pred, u_norm)
        if use_reg:
            loss = loss + model.spectral_regularization()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * a.size(0)
        n += a.size(0)
    return total_loss / n


def evaluate(model, test_loader, criterion, device, u_mean, u_std):
    """Mean relative L2 error over the test set (resolution-independent metric)."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for a, u in test_loader:
            a, u = a.to(device), u.to(device)
            u_norm = (u - u_mean) / (u_std + 1e-8)
            pred = model(a)
            num = torch.linalg.vector_norm((pred - u_norm).reshape(a.size(0), -1), dim=1)
            den = torch.linalg.vector_norm(u_norm.reshape(a.size(0), -1), dim=1) + 1e-8
            total += (num / den).sum().item()
            n += a.size(0)
    return total / n
