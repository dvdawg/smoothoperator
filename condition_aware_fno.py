"""
Condition-Aware Fourier Neural Operator (FNO) with Adaptive Spectral Truncation

This script implements and compares:
1. Standard FNO with low-pass filtering (box filter)
2. Proposed Condition-Aware FNO with adaptive spectral truncation based on utility scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import time


# ============================================================================
# Synthetic Data Generation
# ============================================================================

class PoissonDataset(Dataset):
    """Generate synthetic data for Poisson equation: -Δu = a"""
    
    def __init__(self, n_samples=1000, grid_size=64, seed=42):
        self.n_samples = n_samples
        self.grid_size = grid_size
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Generate random source term a(x,y)
        # Using Gaussian random fields for smoothness
        a = self._generate_random_field(self.grid_size)
        
        # Solve Poisson equation in Fourier space
        u = self._solve_poisson(a)
        
        return torch.FloatTensor(a), torch.FloatTensor(u)
    
    def _generate_random_field(self, size):
        """Generate smooth random field using Gaussian process"""
        # Create frequency domain
        kx = np.fft.fftfreq(size)
        ky = np.fft.fftfreq(size)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)
        
        # Power spectrum (decay with frequency)
        power = 1.0 / (1.0 + K**2)
        power[0, 0] = 0  # Remove DC component
        
        # Generate random field
        phase = np.random.randn(size, size) + 1j * np.random.randn(size, size)
        a_fft = np.sqrt(power) * phase
        a = np.real(np.fft.ifft2(a_fft))
        
        # Normalize
        a = (a - a.mean()) / (a.std() + 1e-8)
        return a
    
    def _solve_poisson(self, a):
        """Solve -Δu = a in Fourier space"""
        a_fft = np.fft.fft2(a)
        kx = np.fft.fftfreq(a.shape[0])
        ky = np.fft.fftfreq(a.shape[1])
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        K2[0, 0] = 1  # Avoid division by zero
        
        u_fft = a_fft / K2
        u = np.real(np.fft.ifft2(u_fft))
        return u


# ============================================================================
# Pre-Analysis: Compute Adaptive Spectral Mask
# ============================================================================

def compute_adaptive_mask(
    a_batch: torch.Tensor,
    u_batch: torch.Tensor,
    modes: int,
    lambda_reg: float = 1e-6
) -> torch.Tensor:
    """
    Compute adaptive spectral mask based on utility scores.
    
    Uses rfft2 format to match SpectralConv2d implementation.
    
    Args:
        a_batch: Input batch [B, H, W]
        u_batch: Output batch [B, H, W]
        modes: Number of modes to keep (equivalent to standard low-pass)
        lambda_reg: Regularization for condition number computation
    
    Returns:
        mask: Boolean tensor [H, W//2+1] for rfft2 format indicating which modes to keep
    """
    device = a_batch.device
    B, H, W = a_batch.shape
    
    # Perform real FFT to match SpectralConv2d format
    a_fft = torch.fft.rfft2(a_batch)  # [B, H, W//2+1] complex
    u_fft = torch.fft.rfft2(u_batch)  # [B, H, W//2+1] complex
    
    H_fft, W_fft = a_fft.shape[1], a_fft.shape[2]
    
    # Initialize score matrix
    scores = torch.zeros(H_fft, W_fft, device=device)
    
    # Process each frequency mode
    for i in range(H_fft):
        for j in range(W_fft):
            # Extract mode k=(i,j) across batch
            X_k = a_fft[:, i, j].unsqueeze(-1)  # [B, 1] complex
            Y_k = u_fft[:, i, j].unsqueeze(-1)  # [B, 1] complex
            
            # Convert to real representation for SVD
            # Stack real and imaginary parts
            X_k_real = torch.stack([X_k.real.squeeze(-1), X_k.imag.squeeze(-1)], dim=-1)  # [B, 2]
            
            # Compute SVD
            try:
                U, S, V = torch.linalg.svd(X_k_real, full_matrices=False)
                # Condition number: σ_max / σ_min
                sigma_max = S[0]
                sigma_min = S[-1] if len(S) > 1 else S[0]
                condition_num = sigma_max / (sigma_min + lambda_reg)
            except:
                # Fallback if SVD fails
                condition_num = torch.tensor(1.0, device=device)
            
            # Compute energy (Frobenius norm) of Y_k
            energy = torch.norm(Y_k, p='fro') ** 2
            
            # Utility score
            score = energy / (condition_num + lambda_reg)
            scores[i, j] = score
    
    # Create mask: select top N modes
    # Flatten and get top-k indices
    scores_flat = scores.flatten()
    n_select = min(modes * modes, len(scores_flat))
    _, top_indices = torch.topk(scores_flat, k=n_select)
    
    # Create boolean mask
    mask = torch.zeros(H_fft * W_fft, dtype=torch.bool, device=device)
    mask[top_indices] = True
    mask = mask.reshape(H_fft, W_fft)
    
    return mask


# ============================================================================
# Ridge Regression Initialization
# ============================================================================

def ridge_regression_init(
    a_batch: torch.Tensor,
    u_batch: torch.Tensor,
    mask: torch.Tensor,
    lambda_reg: float = 1e-4
) -> torch.Tensor:
    """
    Initialize spectral weights using Ridge Regression.
    
    Formula: C_k^H = (X_k^H X_k + λI)^{-1} X_k^H Y_k
    
    Uses rfft2 format to match SpectralConv2d implementation.
    
    Args:
        a_batch: Input batch [B, H, W]
        u_batch: Output batch [B, H, W]
        mask: Boolean mask [H, W//2+1] indicating which modes to initialize
        lambda_reg: Ridge regularization parameter
    
    Returns:
        weights: Complex tensor [H, W//2+1] with initialized weights
    """
    device = a_batch.device
    B, H, W = a_batch.shape
    
    # Perform real FFT to match SpectralConv2d format
    a_fft = torch.fft.rfft2(a_batch)  # [B, H, W//2+1] complex
    u_fft = torch.fft.rfft2(u_batch)  # [B, H, W//2+1] complex
    
    H_fft, W_fft = a_fft.shape[1], a_fft.shape[2]
    
    # Initialize weights
    weights = torch.zeros(H_fft, W_fft, dtype=torch.complex64, device=device)
    
    # For each mode in the mask
    for i in range(H_fft):
        for j in range(W_fft):
            if mask[i, j]:
                X_k = a_fft[:, i, j]  # [B] complex
                Y_k = u_fft[:, i, j]  # [B] complex
                
                # Convert to real representation for matrix operations
                # X_k^H X_k: [B, 1]^H [B, 1] = scalar
                # For complex: X_k^H X_k = |X_k|^2
                XHX = torch.sum(X_k.conj() * X_k).real
                
                # X_k^H Y_k
                XHY = torch.sum(X_k.conj() * Y_k)
                
                # Ridge solution: (X_k^H X_k + λ)^{-1} X_k^H Y_k
                denominator = XHX + lambda_reg
                weight = XHY / (denominator + 1e-10)
                
                weights[i, j] = weight
    
    return weights


# ============================================================================
# Standard Spectral Convolution
# ============================================================================

class SpectralConv2d(nn.Module):
    """Standard 2D Spectral Convolution with low-pass filtering"""
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        # Learnable weights for low frequencies only
        scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
    
    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


# ============================================================================
# Condition-Aware Spectral Convolution
# ============================================================================

class ConditionAwareSpectralConv2d(nn.Module):
    """Condition-Aware 2D Spectral Convolution with adaptive mask"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mask: torch.Tensor,
        modes1: int,
        modes2: int
    ):
        super(ConditionAwareSpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        # Register mask as buffer (not a parameter)
        # Mask is in rfft2 format: [H, W//2+1]
        self.register_buffer('mask', mask)
        
        # Count number of active modes
        n_active = mask.sum().item()
        
        # Learnable weights - same structure as standard FNO
        # We'll mask during forward pass
        scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
    
    def forward(self, x):
        batchsize = x.shape[0]
        H, W = x.size(-2), x.size(-1)
        W_fft = W // 2 + 1
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)  # [B, C, H, W_fft]
        
        # Create output
        out_ft = torch.zeros(
            batchsize, self.out_channels, H, W_fft,
            dtype=torch.cfloat, device=x.device
        )
        
        # Apply mask-based filtering
        # Low frequencies (first modes1 modes)
        low_mask = self.mask[:self.modes1, :self.modes2].unsqueeze(0).unsqueeze(0)  # [1, 1, modes1, modes2]
        masked_weights1 = self.weights1 * low_mask
        out_ft[:, :, :self.modes1, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], masked_weights1)
        
        # High frequencies (last modes1 modes)
        high_mask = self.mask[-self.modes1:, :self.modes2].unsqueeze(0).unsqueeze(0)  # [1, 1, modes1, modes2]
        masked_weights2 = self.weights2 * high_mask
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], masked_weights2)
        
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(H, W))
        return x


# ============================================================================
# FNO Models
# ============================================================================

class FNO2d(nn.Module):
    """Standard FNO with low-pass filtering"""
    
    def __init__(self, modes1=12, modes2=12, width=64):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        self.fc0 = nn.Linear(1, self.width)
        
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class ConditionAwareFNO2d(nn.Module):
    """Condition-Aware FNO with adaptive spectral truncation"""
    
    def __init__(self, mask: torch.Tensor, modes1=12, modes2=12, width=64):
        super(ConditionAwareFNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        self.fc0 = nn.Linear(1, self.width)
        
        self.conv0 = ConditionAwareSpectralConv2d(self.width, self.width, mask, self.modes1, self.modes2)
        self.conv1 = ConditionAwareSpectralConv2d(self.width, self.width, mask, self.modes1, self.modes2)
        self.conv2 = ConditionAwareSpectralConv2d(self.width, self.width, mask, self.modes1, self.modes2)
        self.conv3 = ConditionAwareSpectralConv2d(self.width, self.width, mask, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, u_mean=0.0, u_std=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for a, u in dataloader:
        a = a.to(device).unsqueeze(-1)  # [B, H, W, 1]
        u = u.to(device).unsqueeze(-1)  # [B, H, W, 1]
        # Normalize output
        u = (u - u_mean) / (u_std + 1e-8)
        
        optimizer.zero_grad()
        u_pred = model(a)
        loss = criterion(u_pred, u)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, dataloader, criterion, device, u_mean=0.0, u_std=1.0):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for a, u in dataloader:
            a = a.to(device).unsqueeze(-1)
            u = u.to(device).unsqueeze(-1)
            # Normalize output
            u = (u - u_mean) / (u_std + 1e-8)
            
            u_pred = model(a)
            loss = criterion(u_pred, u)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    grid_size = 64
    modes = 12
    batch_size = 20
    n_train = 800
    n_test = 200
    n_epochs = 50
    learning_rate = 0.001
    
    # Create datasets
    print("Generating synthetic data...")
    train_dataset = PoissonDataset(n_samples=n_train, grid_size=grid_size, seed=42)
    test_dataset = PoissonDataset(n_samples=n_test, grid_size=grid_size, seed=43)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Pre-analysis: Compute adaptive mask and normalization statistics
    print("Computing normalization statistics and adaptive spectral mask...")
    
    # Collect all training data for statistics
    a_batch_list = []
    u_batch_list = []
    for a, u in train_loader:
        a_batch_list.append(a)
        u_batch_list.append(u)
    
    a_all = torch.cat(a_batch_list, dim=0).to(device)
    u_all = torch.cat(u_batch_list, dim=0).to(device)
    
    # Compute normalization statistics from full training data
    u_mean = u_all.mean().item()
    u_std = u_all.std().item()
    print(f"Output statistics: mean={u_mean:.4f}, std={u_std:.4f}")
    print(f"Normalizing outputs to unit scale for training stability...")
    
    # Normalize outputs for mask computation
    u_all_normalized = (u_all - u_mean) / (u_std + 1e-8)
    
    # Use subset for mask computation (can be expensive)
    n_samples_for_mask = min(200, len(a_all))  # Use up to 200 samples
    a_mask = a_all[:n_samples_for_mask]
    u_mask = u_all_normalized[:n_samples_for_mask]
    
    adaptive_mask = compute_adaptive_mask(a_mask, u_mask, modes=modes)
    H_fft, W_fft = adaptive_mask.shape
    print(f"Adaptive mask: {adaptive_mask.sum().item()} modes selected out of {H_fft * W_fft} (rfft2 format)")
    
    # Initialize weights using ridge regression (use normalized data, subset for efficiency)
    print("Computing Ridge Regression initialization...")
    init_weights = ridge_regression_init(a_mask, u_mask, adaptive_mask, lambda_reg=1e-4)
    print(f"Ridge regression weights computed (shape: {init_weights.shape})")
    print("Note: Ridge initialization computed but not applied to all layers (would require channel mapping)")
    
    # Model A: Standard FNO
    print("\n" + "="*60)
    print("Training Standard FNO (Model A)")
    print("="*60)
    model_standard = FNO2d(modes1=modes, modes2=modes, width=64).to(device)
    optimizer_standard = torch.optim.Adam(model_standard.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_losses_standard = []
    test_losses_standard = []
    
    start_time = time.time()
    for epoch in range(n_epochs):
        train_loss = train_epoch(model_standard, train_loader, optimizer_standard, criterion, device, u_mean, u_std)
        test_loss = evaluate(model_standard, test_loader, criterion, device, u_mean, u_std)
        
        train_losses_standard.append(train_loss)
        test_losses_standard.append(test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    standard_time = time.time() - start_time
    print(f"Standard FNO training completed in {standard_time:.2f} seconds")
    print(f"Final Test Loss: {test_losses_standard[-1]:.6f}")
    
    # Model B: Condition-Aware FNO
    print("\n" + "="*60)
    print("Training Condition-Aware FNO (Model B)")
    print("="*60)
    model_adaptive = ConditionAwareFNO2d(
        mask=adaptive_mask, modes1=modes, modes2=modes, width=64
    ).to(device)
    
    # Initialize first spectral layer with ridge regression weights
    # Map the single-channel init_weights to the multi-channel layer
    with torch.no_grad():
        # Initialize weights1 (low frequencies) - broadcast across channels
        low_mask = adaptive_mask[:modes, :modes]
        if low_mask.any():
            init_low = init_weights[:modes, :modes]  # [modes, modes]
            # Broadcast to [in_channels, out_channels, modes, modes]
            for i in range(model_adaptive.conv0.in_channels):
                for o in range(model_adaptive.conv0.out_channels):
                    # Scale by a small factor to account for multi-channel
                    scale_factor = 1.0 / np.sqrt(model_adaptive.conv0.in_channels * model_adaptive.conv0.out_channels)
                    model_adaptive.conv0.weights1.data[i, o] = init_low * scale_factor
        
        # Initialize weights2 (high frequencies)
        high_mask = adaptive_mask[-modes:, :modes]
        if high_mask.any():
            init_high = init_weights[-modes:, :modes]  # [modes, modes]
            for i in range(model_adaptive.conv0.in_channels):
                for o in range(model_adaptive.conv0.out_channels):
                    scale_factor = 1.0 / np.sqrt(model_adaptive.conv0.in_channels * model_adaptive.conv0.out_channels)
                    model_adaptive.conv0.weights2.data[i, o] = init_high * scale_factor
    
    print("First spectral layer initialized with Ridge Regression weights")
    optimizer_adaptive = torch.optim.Adam(model_adaptive.parameters(), lr=learning_rate)
    
    train_losses_adaptive = []
    test_losses_adaptive = []
    
    start_time = time.time()
    for epoch in range(n_epochs):
        train_loss = train_epoch(model_adaptive, train_loader, optimizer_adaptive, criterion, device, u_mean, u_std)
        test_loss = evaluate(model_adaptive, test_loader, criterion, device, u_mean, u_std)
        
        train_losses_adaptive.append(train_loss)
        test_losses_adaptive.append(test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    adaptive_time = time.time() - start_time
    print(f"Condition-Aware FNO training completed in {adaptive_time:.2f} seconds")
    print(f"Final Test Loss: {test_losses_adaptive[-1]:.6f}")
    
    # Results Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Standard FNO:")
    print(f"  Final Test Loss: {test_losses_standard[-1]:.6f}")
    print(f"  Training Time: {standard_time:.2f}s")
    print(f"\nCondition-Aware FNO:")
    print(f"  Final Test Loss: {test_losses_adaptive[-1]:.6f}")
    print(f"  Training Time: {adaptive_time:.2f}s")
    print(f"  Improvement: {((test_losses_standard[-1] - test_losses_adaptive[-1]) / test_losses_standard[-1] * 100):.2f}%")
    
    # Plot convergence
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_standard, label='Standard FNO (Train)', linestyle='--')
    plt.plot(test_losses_standard, label='Standard FNO (Test)', linestyle='-')
    plt.plot(train_losses_adaptive, label='Condition-Aware FNO (Train)', linestyle='--')
    plt.plot(test_losses_adaptive, label='Condition-Aware FNO (Test)', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Convergence')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(test_losses_standard, label='Standard FNO', linewidth=2)
    plt.plot(test_losses_adaptive, label='Condition-Aware FNO', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Test MSE Loss')
    plt.title('Test Loss Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fno_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to 'fno_comparison.png'")


if __name__ == "__main__":
    main()

