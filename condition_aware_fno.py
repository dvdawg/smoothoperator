import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def compute_adaptive_mask(
    a_batch: torch.Tensor,
    u_batch: torch.Tensor,
    modes1: int,
    modes2: int,
    energy_fraction: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the adaptive spectral mask using the Condition-Aware criterion:
    Score(k) = Energy(k) / ConditionNumber(k).
    
    Modes are selected by sorting Score(k) descending and retaining the top set 
    that captures 'energy_fraction' of the total energy[cite: 143, 146].
    """
    device = a_batch.device

    # Ensure inputs are (B, C, H, W)
    if a_batch.dim() == 3:
        a_batch_in = a_batch.unsqueeze(1)
    else:
        a_batch_in = a_batch

    if u_batch.dim() == 3:
        u_batch_in = u_batch.unsqueeze(1)
    else:
        u_batch_in = u_batch

    # Compute FFTs: (B, C, H, W_half)
    a_fft = torch.fft.rfft2(a_batch_in)
    u_fft = torch.fft.rfft2(u_batch_in)
    
    _, _, H_fft, W_fft = a_fft.shape
    m2_eff = min(modes2, W_fft)

    def compute_metrics(i_start: int, i_end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        m1 = max(i_end - i_start, 0)
        scores = torch.zeros((m1, modes2), device=device)
        energies = torch.zeros((m1, modes2), device=device)
        
        for i_abs in range(i_start, i_end):
            i_rel = i_abs - i_start
            for j in range(m2_eff):
                # Extract mode k data across batch: (Batch, In_Channels)
                X_k = a_fft[:, :, i_abs, j]
                Y_k = u_fft[:, :, i_abs, j]
                
                # 1. Compute Energy
                energy = torch.sum(torch.abs(Y_k) ** 2).real
                energies[i_rel, j] = energy
                
                # 2. Compute Condition Number of Input (X_k) [cite: 130]
                # kappa = sigma_max / sigma_min
                if X_k.shape[1] > 0:
                    try:
                        # Compute singular values (S is sorted desc)
                        s = torch.linalg.svdvals(X_k)
                        if s.numel() > 0 and s[-1] > 1e-9:
                            cond = s[0] / s[-1]
                        else:
                            cond = 1e9  # Ill-conditioned or zero signal
                    except:
                        cond = 1e9
                else:
                    cond = 1.0

                # 3. Compute Score = Energy / Condition [cite: 143]
                scores[i_rel, j] = energy / (cond + 1e-8)
                
        return scores, energies

    def select_mask(scores: torch.Tensor, energies: torch.Tensor) -> torch.Tensor:
        flat_scores = scores.reshape(-1)
        flat_energies = energies.reshape(-1)
        
        total_energy = flat_energies.sum()
        if total_energy <= 0:
            return torch.ones_like(scores, dtype=torch.bool, device=device)
            
        # Sort by utility score [cite: 144]
        sorted_scores, sorted_idx = torch.sort(flat_scores, descending=True)
        
        # Cumulative energy selection [cite: 146]
        sorted_energies = flat_energies[sorted_idx]
        cumsum_energy = torch.cumsum(sorted_energies, dim=0)
        
        target = energy_fraction * total_energy
        k = int((cumsum_energy <= target).sum().item())
        
        # Keep at least one mode if energy exists
        if k == 0 and total_energy > 0:
            k = 1
        if k > flat_scores.numel():
            k = flat_scores.numel()

        mask_flat = torch.zeros_like(flat_scores, dtype=torch.bool, device=device)
        mask_flat[sorted_idx[:k]] = True
        
        return mask_flat.view_as(scores)

    # Process "top-left" corner of frequencies (positive modes)
    low_start = 0
    low_end = min(modes1, H_fft)
    
    # Process "bottom-left" corner of frequencies (negative modes via aliasing)
    high_end = H_fft
    high_start = max(high_end - modes1, 0)

    low_scores, low_energies = compute_metrics(low_start, low_end)
    high_scores, high_energies = compute_metrics(high_start, high_end)

    low_mask = select_mask(low_scores, low_energies)
    high_mask = select_mask(high_scores, high_energies)
    
    return low_mask, high_mask


def ridge_regression_init(
    a_batch: torch.Tensor,
    u_batch: torch.Tensor,
    low_mask: torch.Tensor,
    high_mask: torch.Tensor,
    modes1: int,
    modes2: int,
    in_channels: int,
    out_channels: int,
    lambda_reg: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fits the spectral weights C_k using Ridge Regression on the training batch[cite: 121, 137].
    Solves C_k^H = (X_k^H X_k + lambda I)^-1 X_k^H Y_k[cite: 139].
    """
    device = a_batch.device

    # Ensure inputs are (B, C, H, W)
    if a_batch.dim() == 3: a_batch = a_batch.unsqueeze(1)
    if u_batch.dim() == 3: u_batch = u_batch.unsqueeze(1)

    a_fft = torch.fft.rfft2(a_batch)
    u_fft = torch.fft.rfft2(u_batch)
    _, _, H_fft, W_fft = a_fft.shape
    m2_eff = min(modes2, W_fft)

    def init_region(mask: torch.Tensor, i_start: int, i_end: int) -> torch.Tensor:
        m1 = max(i_end - i_start, 0)
        # Weights shape: (in, out, modes1, modes2)
        weights = torch.zeros((in_channels, out_channels, m1, modes2), dtype=torch.cfloat, device=device)
        
        for i_abs in range(i_start, i_end):
            i_rel = i_abs - i_start
            for j in range(m2_eff):
                if i_rel < mask.shape[0] and j < mask.shape[1] and mask[i_rel, j]:
                    # Extract Data Matrices
                    # X_k: (Batch, In_Channels)
                    # Y_k: (Batch, Out_Channels)
                    X_k = a_fft[:, :, i_abs, j]
                    Y_k = u_fft[:, :, i_abs, j]
                    
                    # Ridge Regression Closed Form: C_k^H = (X^H X + lambda I)^-1 X^H Y
                    # Note on dimensions:
                    # We want weights W of shape (In, Out) such that Y ~ X @ W.
                    
                    # 1. Compute X^H X (In, In)
                    XHX = torch.matmul(X_k.T.conj(), X_k)
                    
                    # 2. Add regularization
                    reg = lambda_reg * torch.eye(in_channels, device=device, dtype=XHX.dtype)
                    
                    # 3. Compute X^H Y (In, Out)
                    XHY = torch.matmul(X_k.T.conj(), Y_k)
                    
                    # 4. Solve for W
                    # W = (XHX + reg)^-1 @ XHY
                    try:
                        w_sol = torch.linalg.solve(XHX + reg, XHY)
                    except RuntimeError:
                        # Fallback for numerical instability
                        w_sol = torch.zeros((in_channels, out_channels), dtype=torch.cfloat, device=device)
                    
                    weights[:, :, i_rel, j] = w_sol
                    
        return weights

    low_start = 0
    low_end = min(modes1, H_fft)
    high_end = H_fft
    high_start = max(high_end - modes1, 0)

    low_weights = init_region(low_mask, low_start, low_end)
    high_weights = init_region(high_mask, high_start, high_end)
    
    return low_weights, high_weights


################################################################
# 2. Model Architecture
################################################################

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
        init_weights1: torch.Tensor = None,
        init_weights2: torch.Tensor = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Register masks as buffers so they are saved with state_dict but not optimized
        self.register_buffer("low_mask", low_mask.bool())
        self.register_buffer("high_mask", high_mask.bool())

        # Initialize weights
        scale = 1.0 / (in_channels * out_channels)
        
        # If passed Ridge Regression weights, use them. Otherwise random init.
        if init_weights1 is not None:
            self.weights1 = nn.Parameter(init_weights1.clone())
        else:
            self.weights1 = nn.Parameter(
                scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
            )
            
        if init_weights2 is not None:
            self.weights2 = nn.Parameter(init_weights2.clone())
        else:
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

        # Apply mask to weights during forward pass
        # This ensures pruned modes remain zero even if optimizer updates them
        low_mask_active = self.low_mask[:m1, :m2].to(torch.cfloat)
        high_mask_active = self.high_mask[:m1, :m2].to(torch.cfloat)

        w1 = self.weights1[:, :, :m1, :m2] * low_mask_active.unsqueeze(0).unsqueeze(0)
        w2 = self.weights2[:, :, :m1, :m2] * high_mask_active.unsqueeze(0).unsqueeze(0)

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
        if x.dim() == 3:
            x = x.unsqueeze(-1)
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
        x = x.squeeze(-1)
        return x


class ConditionAwareFNO2d(nn.Module):
    def __init__(
        self,
        low_mask: torch.Tensor,
        high_mask: torch.Tensor,
        low_weights: torch.Tensor,
        high_weights: torch.Tensor,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = nn.Linear(1, width)

        # The first layer is the Condition-Aware layer
        # It takes the masks and the Ridge Regression initialized weights
        self.conv0 = ConditionAwareSpectralConv2d(
            width, width, 
            low_mask, high_mask, 
            modes1, modes2,
            init_weights1=low_weights,
            init_weights2=high_weights
        )
        
        # Subsequent layers are standard FNO layers
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
        x = x.squeeze(-1)
        return x


################################################################
# 3. Training / Evaluation Loop
################################################################

def train_epoch(model, train_loader, optimizer, criterion, device, u_mean, u_std):
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    for a, u in train_loader:
        a = a.to(device)
        u = u.to(device)
        
        # Normalize target
        u_normalized = (u - u_mean) / (u_std + 1e-8)
        
        optimizer.zero_grad()
        u_pred = model(a)
        
        loss = criterion(u_pred, u_normalized)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * a.size(0)
        n_samples += a.size(0)
    
    return total_loss / n_samples


def evaluate(model, test_loader, criterion, device, u_mean, u_std):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for a, u in test_loader:
            a = a.to(device)
            u = u.to(device)
            
            u_normalized = (u - u_mean) / (u_std + 1e-8)
            
            u_pred = model(a)
            
            loss = criterion(u_pred, u_normalized)
            
            total_loss += loss.item() * a.size(0)
            n_samples += a.size(0)
    
    return total_loss / n_samples