from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from condition_aware_fno import (
    ConditionAwareFNO2d,
    ConditionAwareSpectralConv2d,
    SpectralConv2d,
    compute_adaptive_mask,
    evaluate,
    train_epoch,
    _energy_cond,
    _select_by_energy_budget,
    _to_channels_first,
)

# backwards-compatible local alias
_energy_and_cond = _energy_cond


def _fft_data_matrices(
    a_batch: torch.Tensor,
    u_batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """rfft2 of channels-last (or channels-first) input/target fields."""
    return torch.fft.rfft2(_to_channels_first(a_batch)), torch.fft.rfft2(_to_channels_first(u_batch))


def _build_mask_from_scores(
    scores: torch.Tensor,
    energies: torch.Tensor,
    energy_fraction: float,
    device: torch.device = None,
) -> torch.Tensor:
    """Smallest high-utility set whose cumulative energy meets the budget."""
    return _select_by_energy_budget(scores, energies, energy_fraction)

def compute_learnable_lambda(
    a_train: torch.Tensor,
    u_train: torch.Tensor,
    a_val: torch.Tensor,
    u_val: torch.Tensor,
    modes1: int,
    modes2: int,
    n_iters: int = 60,
    outer_lr: float = 0.05,
    energy_fraction: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = a_train.device

    a_fft_tr, u_fft_tr = _fft_data_matrices(a_train, u_train)
    a_fft_val, u_fft_val = _fft_data_matrices(a_val, u_val)

    _, _, H_fft, W_fft = a_fft_tr.shape
    m2_eff = min(modes2, W_fft)

    def _optimise_region(
        i_start: int, i_end: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m1 = max(i_end - i_start, 0)
        lambdas = torch.zeros((m1, modes2), device=device)
        scores = torch.zeros((m1, modes2), device=device)
        energies = torch.zeros((m1, modes2), device=device)

        for i_abs in range(i_start, i_end):
            i_rel = i_abs - i_start
            for j in range(m2_eff):
                X_tr = a_fft_tr[:, :, i_abs, j]                 
                Y_tr = u_fft_tr[:, :, i_abs, j]                  
                X_val = a_fft_val[:, :, i_abs, j]
                Y_val = u_fft_val[:, :, i_abs, j]

                d_in = X_tr.shape[1]

                XHX = X_tr.T.conj() @ X_tr                       # (d, d)
                B_mat = X_tr.T.conj() @ Y_tr                     # (d, o)
                eye = torch.eye(d_in, dtype=XHX.dtype, device=device)

                # Bilevel descent on alpha = log(lambda) using the *analytic*
                # gradient of the closed-form ridge solution (paper Eq. for
                # d L_val / d alpha), avoiding autograd through linalg.solve.
                #   A = XHX + lam I,  C* = A^{-1} B,
                #   dC*/dalpha = -lam A^{-1} C*,
                #   dL/dalpha  = 2 Re < X_val dC*/dalpha , X_val C* - Y_val >.
                log_lam = 0.0
                for _ in range(n_iters):
                    lam = torch.exp(torch.tensor(log_lam, dtype=torch.float64)).item()
                    A = XHX + lam * eye
                    try:
                        Ainv = torch.linalg.inv(A)
                    except torch.linalg.LinAlgError:
                        break
                    C_star = Ainv @ B_mat                        # (d, o)
                    R = X_val @ C_star - Y_val                   # (Nval, o)
                    dC = -lam * (Ainv @ C_star)                  # dC*/dalpha
                    grad = 2.0 * torch.sum((X_val @ dC).conj() * R).real.item()
                    log_lam -= outer_lr * grad
                    log_lam = float(max(min(log_lam, 12.0), -12.0))  # clamp

                lambdas[i_rel, j] = float(torch.exp(torch.tensor(log_lam)).item())

                                                  
                energy, cond = _energy_and_cond(X_tr, Y_tr)
                energies[i_rel, j] = energy
                scores[i_rel, j] = energy / (cond + 1e-8)

        return lambdas, scores, energies

    low_start = 0
    low_end = min(modes1, H_fft)
    high_end = H_fft
    high_start = max(high_end - modes1, 0)

    low_lam, low_scores, low_energies = _optimise_region(low_start, low_end)
    high_lam, high_scores, high_energies = _optimise_region(high_start, high_end)

    low_mask = _build_mask_from_scores(low_scores, low_energies, energy_fraction, device)
    high_mask = _build_mask_from_scores(high_scores, high_energies, energy_fraction, device)

    return low_lam, high_lam, low_mask, high_mask

def _normalized_lambda(low_lam, high_lam, base_scale):
    """Map raw per-mode bilevel lambdas to gentle per-mode weight-decay coefficients.

    The bilevel lambdas live on the physical-input ridge problem and are only
    meaningful *relatively* (larger for ill-conditioned modes).  We normalise to
    unit mean and scale by ``base_scale`` so the spectral weight penalty stays a
    mild, mode-adaptive prior rather than dominating the data term.
    """
    out = []
    for lam in (low_lam, high_lam):
        m = lam.mean()
        norm = lam / (m + 1e-12) if m > 0 else torch.ones_like(lam)
        out.append(base_scale * norm)
    return out


class LearnableLambdaCAFNO2d(nn.Module):
    """CA-FNO whose spectral weights carry a learned, mode-adaptive Tikhonov prior.

    The mask is applied at every layer (like ``ConditionAwareFNO2d``) and each
    layer's spectral weights are penalised by ``sum_k lambda_k ||W(k)||^2`` with
    the per-mode ``lambda_k`` obtained from the bilevel pre-analysis.  This is
    the variant for which the Sec. 7 conditioning argument applies to the
    *trained* weights.
    """

    def __init__(
        self,
        low_mask: torch.Tensor,
        high_mask: torch.Tensor,
        low_lam: torch.Tensor = None,
        high_lam: torch.Tensor = None,
        in_channels: int = 1,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
        reg_scale: float = 1e-3,
    ):
        super().__init__()
        self.modes1, self.modes2, self.width = modes1, modes2, width
        self.fc0 = nn.Linear(in_channels, width)

        if low_lam is None:
            low_lam = torch.ones(modes1, modes2)
        if high_lam is None:
            high_lam = torch.ones(modes1, modes2)
        ll, hl = _normalized_lambda(low_lam, high_lam, reg_scale)

        def _mk():
            return ConditionAwareSpectralConv2d(
                width, width, low_mask, high_mask, modes1, modes2, ll, hl
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

class PerLayerCAFNO2d(nn.Module):

    _N_LAYERS: int = 4

    def __init__(
        self,
        low_mask: torch.Tensor,
        high_mask: torch.Tensor,
        in_channels: int = 1,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = nn.Linear(in_channels, width)



        self.conv0 = ConditionAwareSpectralConv2d(
            width, width, low_mask, high_mask, modes1, modes2
        )
        self.conv1 = ConditionAwareSpectralConv2d(
            width, width, low_mask, high_mask, modes1, modes2
        )
        self.conv2 = ConditionAwareSpectralConv2d(
            width, width, low_mask, high_mask, modes1, modes2
        )
        self.conv3 = ConditionAwareSpectralConv2d(
            width, width, low_mask, high_mask, modes1, modes2
        )

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

        self._convs = [self.conv0, self.conv1, self.conv2, self.conv3]

                                                                               

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x).squeeze(-1)
        return x

                                                                               

    @torch.no_grad()
    def bootstrap_layer_masks(
        self,
        train_loader,
        device: torch.device,
        energy_fraction: float = 0.95,
        n_batches: int = 20,
    ) -> None:
        self.eval()
        self.to(device)

                                                                            
        activations: Dict[int, List[torch.Tensor]] = {i: [] for i in range(self._N_LAYERS + 1)}

        def _make_hook(layer_idx: int):
            def _hook(module, inp, out):
                                                                      
                activations[layer_idx].append(inp[0].detach().cpu())
                if layer_idx == self._N_LAYERS - 1:
                    activations[self._N_LAYERS].append(out.detach().cpu())
            return _hook

        handles = []
        for idx, conv in enumerate(self._convs):
            handles.append(conv.register_forward_hook(_make_hook(idx)))

                                                                  
        for batch_idx, (a, _) in enumerate(train_loader):
            if batch_idx >= n_batches:
                break
            a = a.to(device)
            if a.dim() == 3:
                a = a.unsqueeze(-1)
                                                                
            h = self.fc0(a)
            h = h.permute(0, 3, 1, 2)
                                                                
            _ = self.forward(a.squeeze(-1))

        for h in handles:
            h.remove()

                                                                            
        for layer_idx in range(self._N_LAYERS):
            feats_in = torch.cat(activations[layer_idx], dim=0)                    
            feats_out = torch.cat(activations[layer_idx + 1], dim=0)               

            feats_in = feats_in.to(device)
            feats_out = feats_out.to(device)

            low_mask, high_mask = self._score_layer_transition(
                feats_in, feats_out, energy_fraction
            )

                                                                 
            new_conv = ConditionAwareSpectralConv2d(
                self.width, self.width,
                low_mask, high_mask,
                self.modes1, self.modes2,
            ).to(device)

                                                                              
            old_conv = self._convs[layer_idx]
            new_conv.weights1.data.copy_(old_conv.weights1.data)
            new_conv.weights2.data.copy_(old_conv.weights2.data)

            setattr(self, f"conv{layer_idx}", new_conv)
            self._convs[layer_idx] = new_conv

            n_low = low_mask.sum().item()
            n_high = high_mask.sum().item()
            total = self.modes1 * self.modes2
            print(
                f"  Layer {layer_idx}: low={n_low}/{total}, high={n_high}/{total} modes retained"
            )

    def _score_layer_transition(
        self,
        feats_in: torch.Tensor,
        feats_out: torch.Tensor,
        energy_fraction: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score the mode transition between two consecutive feature fields."""
        device = feats_in.device
        a_fft = torch.fft.rfft2(feats_in)                             
        u_fft = torch.fft.rfft2(feats_out)

        _, _, H_fft, W_fft = a_fft.shape
        m2_eff = min(self.modes2, W_fft)

        def _metrics(i_start, i_end):
            m1 = max(i_end - i_start, 0)
            scores = torch.zeros((m1, self.modes2), device=device)
            energies = torch.zeros((m1, self.modes2), device=device)
            for i_abs in range(i_start, i_end):
                i_rel = i_abs - i_start
                for j in range(m2_eff):
                    X_k = a_fft[:, :, i_abs, j]              
                    Y_k = u_fft[:, :, i_abs, j]
                    energy, cond = _energy_and_cond(X_k, Y_k)
                    energies[i_rel, j] = energy
                    scores[i_rel, j] = energy / (cond + 1e-8)
            return scores, energies

        low_start, low_end = 0, min(self.modes1, H_fft)
        high_end = H_fft
        high_start = max(high_end - self.modes1, 0)

        low_s, low_e = _metrics(low_start, low_end)
        high_s, high_e = _metrics(high_start, high_end)

        return (
            _build_mask_from_scores(low_s, low_e, energy_fraction, device),
            _build_mask_from_scores(high_s, high_e, energy_fraction, device),
        )

                                                                                
                                           
                                                                                

class DynamicCAFNO2d(nn.Module):
    def __init__(
        self,
        low_mask: torch.Tensor,
        high_mask: torch.Tensor,
        in_channels: int = 1,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = nn.Linear(in_channels, width)

        def _mk():
            return ConditionAwareSpectralConv2d(
                width, width, low_mask, high_mask, modes1, modes2
            )

        self.conv0, self.conv1, self.conv2, self.conv3 = _mk(), _mk(), _mk(), _mk()
        self._convs = [self.conv0, self.conv1, self.conv2, self.conv3]

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

    @torch.no_grad()
    def update_masks_from_residuals(
        self,
        a_batch: torch.Tensor,
        u_batch: torch.Tensor,
        u_mean: float,
        u_std: float,
        energy_fraction: float = 0.95,
    ) -> Tuple[int, int]:
        self.eval()
        device = a_batch.device

                                                     
        pred = self.forward(a_batch)                                              
        u_norm = (u_batch - u_mean) / (u_std + 1e-8)
        residual = u_norm - pred                                      

                                                                      
        a_fft, r_fft = _fft_data_matrices(a_batch, residual)
        _, _, H_fft, W_fft = a_fft.shape
        m2_eff = min(self.modes2, W_fft)

        def _residual_metrics(i_start: int, i_end: int):
            m1 = max(i_end - i_start, 0)
            scores = torch.zeros((m1, self.modes2), device=device)
            energies = torch.zeros((m1, self.modes2), device=device)
            for i_abs in range(i_start, i_end):
                i_rel = i_abs - i_start
                for j in range(m2_eff):
                    X_k = a_fft[:, :, i_abs, j]
                    R_k = r_fft[:, :, i_abs, j]
                    energy, cond = _energy_and_cond(X_k, R_k)
                    energies[i_rel, j] = energy
                    scores[i_rel, j] = energy / (cond + 1e-8)
            return scores, energies

        low_start, low_end = 0, min(self.modes1, H_fft)
        high_end = H_fft
        high_start = max(high_end - self.modes1, 0)

        low_s, low_e = _residual_metrics(low_start, low_end)
        high_s, high_e = _residual_metrics(high_start, high_end)

        new_low_mask = _build_mask_from_scores(low_s, low_e, energy_fraction, device)
        new_high_mask = _build_mask_from_scores(high_s, high_e, energy_fraction, device)

        # update the active mode set at every spectral layer
        for conv in self._convs:
            conv.low_mask.copy_(new_low_mask)
            conv.high_mask.copy_(new_high_mask)

        return new_low_mask.sum().item(), new_high_mask.sum().item()

def geometric_schedule(n_epochs: int, n_updates: int = 4, rho: float = 0.6) -> List[int]:

    if n_updates <= 0:
        return []
    schedule = []
    for i in range(n_updates):
        t = int(n_epochs * rho ** (n_updates - 1 - i))
        t = max(t, 1)
        schedule.append(t)
                           
    return sorted(set(schedule))

def train_with_dynamic_updates(
    model: DynamicCAFNO2d,
    train_loader,
    test_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    u_mean: float,
    u_std: float,
    n_epochs: int,
    update_schedule: Optional[List[int]] = None,
    energy_fraction: float = 0.95,
    n_batches_for_update: int = 20,
) -> Dict:

    if update_schedule is None:
        update_schedule = geometric_schedule(n_epochs)

    schedule_set = set(update_schedule)
    train_losses, test_losses = [], []
    update_events = []                                        

                                                                            
    a_update_list, u_update_list = [], []
    for batch_idx, (a, u) in enumerate(train_loader):
        if batch_idx >= n_batches_for_update:
            break
        a_update_list.append(a)
        u_update_list.append(u)
    a_update = torch.cat(a_update_list, dim=0).to(device)
    u_update = torch.cat(u_update_list, dim=0).to(device)

    start = time.time()
    for epoch in range(n_epochs):
                                                                           
        if epoch in schedule_set:
            n_low, n_high = model.update_masks_from_residuals(
                a_update, u_update, u_mean, u_std, energy_fraction
            )
            update_events.append((epoch, n_low, n_high))
            print(
                f"  [Dynamic] Epoch {epoch}: mask updated → "
                f"low={n_low}, high={n_high} modes active"
            )

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, u_mean, u_std
        )
        test_loss = evaluate(model, test_loader, criterion, device, u_mean, u_std)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1}/{n_epochs} — "
                f"Train: {train_loss:.6f}  Test: {test_loss:.6f}"
            )

    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "final_test_loss": test_losses[-1],
        "training_time": time.time() - start,
        "update_events": update_events,
    }                                                                     

def compute_full_spectrum_mask(
    a_batch: torch.Tensor,
    u_batch: torch.Tensor,
    H_fft: int,
    W_fft: int,
    energy_fraction: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:

    device = a_batch.device
    a_fft, u_fft = _fft_data_matrices(a_batch, u_batch)

    half = H_fft // 2

    def _score_block(i_start: int, i_end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        m1 = i_end - i_start
        scores = torch.zeros((m1, W_fft), device=device)
        energies = torch.zeros((m1, W_fft), device=device)
        for i_abs in range(i_start, i_end):
            i_rel = i_abs - i_start
            for j in range(W_fft):
                X_k = a_fft[:, :, i_abs, j]
                Y_k = u_fft[:, :, i_abs, j]
                energy, cond = _energy_and_cond(X_k, Y_k)
                energies[i_rel, j] = energy
                scores[i_rel, j] = energy / (cond + 1e-8)
        return scores, energies

    low_s, low_e = _score_block(0, half)
    high_s, high_e = _score_block(H_fft - half, H_fft)

    low_mask = _build_mask_from_scores(low_s, low_e, energy_fraction, device)
    high_mask = _build_mask_from_scores(high_s, high_e, energy_fraction, device)

    return low_mask, high_mask

class AnisotropicSpectralConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        low_mask: torch.Tensor,                       
        high_mask: torch.Tensor,                      
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        half, W_fft = low_mask.shape
        self.half = half
        self.W_fft = W_fft

        self.register_buffer("low_mask", low_mask.bool())
        self.register_buffer("high_mask", high_mask.bool())

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, half, W_fft, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, half, W_fft, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)                                           
        B, C, H_fft, W_fft_act = x_ft.shape

        half = min(self.half, H_fft // 2)
        wf = min(self.W_fft, W_fft_act)

        out_ft = torch.zeros(
            B, self.out_channels, H_fft, W_fft_act, dtype=torch.cfloat, device=x.device
        )

        low_m = self.low_mask[:half, :wf].to(torch.cfloat)
        high_m = self.high_mask[:half, :wf].to(torch.cfloat)

        w1 = self.weights1[:, :, :half, :wf] * low_m.unsqueeze(0).unsqueeze(0)
        w2 = self.weights2[:, :, :half, :wf] * high_m.unsqueeze(0).unsqueeze(0)

        out_ft[:, :, :half, :wf] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :half, :wf], w1
        )
        out_ft[:, :, -half:, :wf] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -half:, :wf], w2
        )

        return torch.fft.irfft2(out_ft, s=(H, W))

class AnisotropicCAFNO2d(nn.Module):

    def __init__(
        self,
        low_mask: torch.Tensor,
        high_mask: torch.Tensor,
        in_channels: int = 1,
        width: int = 64,
    ):
        super().__init__()
        self.width = width

        half, W_fft = low_mask.shape
        self.half = half
        self.W_fft = W_fft

        self.fc0 = nn.Linear(in_channels, width)

        self.conv0 = AnisotropicSpectralConv2d(width, width, low_mask, high_mask)
        self.conv1 = AnisotropicSpectralConv2d(width, width, low_mask, high_mask)
        self.conv2 = AnisotropicSpectralConv2d(width, width, low_mask, high_mask)
        self.conv3 = AnisotropicSpectralConv2d(width, width, low_mask, high_mask)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x).squeeze(-1)
        return x                                                                              

def mask_anisotropy_stats(
    low_mask: torch.Tensor,
    high_mask: torch.Tensor,
) -> Dict:
    low = low_mask.float()
    H, W = low.shape

    row_density = low.mean(dim=1)                                              
    col_density = low.mean(dim=0)                                              
    row_spread = row_density.std().item()
    col_spread = col_density.std().item()

    return {
        "frac_low": low_mask.float().mean().item(),
        "frac_high": high_mask.float().mean().item(),
        "row_density": row_density.cpu().tolist(),
        "col_density": col_density.cpu().tolist(),
        "aspect_ratio": (row_spread / col_spread) if col_spread > 1e-9 else float("inf"),
    }