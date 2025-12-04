"""
Dataset generators for testing Condition-Aware FNO

Includes multiple PDE datasets:
- Poisson equation
- Darcy flow
- Burgers' equation (1D -> 2D)
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple


class BasePDEDataset(Dataset):
    """Base class for PDE datasets"""
    
    def __init__(self, n_samples=1000, grid_size=64, seed=42):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.seed = seed
        
        # Generate all data upfront for reproducibility
        # This ensures the same data is returned regardless of access order
        self._generate_all_data()
    
    def _generate_all_data(self):
        """Generate all data samples upfront - override in subclasses"""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.data = []
        for idx in range(self.n_samples):
            # Use idx as part of seed to ensure each sample is deterministic
            sample_seed = self.seed + idx * 1000
            np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            
            a = self._generate_random_field(self.grid_size)
            u = self._solve_pde(a)
            self.data.append((torch.FloatTensor(a), torch.FloatTensor(u)))
    
    def _solve_pde(self, a):
        """Override in subclasses to implement specific PDE solver"""
        raise NotImplementedError("Subclasses must implement _solve_pde")
    
    def _generate_random_field(self, size, power_decay=1.0):
        """Generate smooth random field using Gaussian process"""
        # Create frequency domain
        kx = np.fft.fftfreq(size)
        ky = np.fft.fftfreq(size)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)
        
        # Power spectrum (decay with frequency)
        power = 1.0 / (1.0 + K**power_decay)
        power[0, 0] = 0  # Remove DC component
        
        # Generate random field
        phase = np.random.randn(size, size) + 1j * np.random.randn(size, size)
        a_fft = np.sqrt(power) * phase
        a = np.real(np.fft.ifft2(a_fft))
        
        # Normalize
        a = (a - a.mean()) / (a.std() + 1e-8)
        return a


class PoissonDataset(BasePDEDataset):
    """Generate synthetic data for Poisson equation: -Δu = a"""
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _solve_pde(self, a):
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


class DarcyDataset(BasePDEDataset):
    """
    Generate synthetic data for Darcy flow equation:
    -∇·(a(x,y)∇u) = f
    
    Simplified version: solve for u given permeability field a
    """
    
    def __init__(self, n_samples=1000, grid_size=64, seed=42, permeability_range=(0.1, 10.0)):
        super().__init__(n_samples, grid_size, seed)
        self.permeability_range = permeability_range
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _generate_all_data(self):
        """Generate all data samples upfront"""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.data = []
        for idx in range(self.n_samples):
            sample_seed = self.seed + idx * 1000
            np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            
            # Generate permeability field a(x,y)
            a_log = self._generate_random_field(self.grid_size, power_decay=2.0)
            # Map to permeability range
            a_min, a_max = self.permeability_range
            a = np.exp(a_log * np.log(a_max / a_min) / 2 + np.log((a_min * a_max) ** 0.5))
            
            # Generate source term f
            f = self._generate_random_field(self.grid_size, power_decay=1.5)
            f = (f - f.min()) / (f.max() - f.min() + 1e-8) * 2 - 1  # Normalize to [-1, 1]
            
            # Solve Darcy equation: -∇·(a∇u) = f
            u = self._solve_pde(a, f)
            
            self.data.append((torch.FloatTensor(a), torch.FloatTensor(u)))
    
    def _solve_pde(self, a, f):
        """Solve -∇·(a∇u) = f using iterative method"""
        # Simplified: use Fourier method with constant coefficient approximation
        # More accurate would use finite difference, but this is faster for synthetic data
        dx = 1.0 / self.grid_size
        
        # Use a simplified approach: solve in Fourier space with averaged coefficient
        a_avg = a.mean()
        f_fft = np.fft.fft2(f)
        kx = np.fft.fftfreq(self.grid_size)
        ky = np.fft.fftfreq(self.grid_size)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        K2[0, 0] = 1
        
        # Approximate solution
        u_fft = f_fft / (a_avg * K2 + 1e-6)
        u = np.real(np.fft.ifft2(u_fft))
        
        # Add correction for spatially varying coefficient (simplified)
        # This is a heuristic to make the problem more interesting
        u = u * (1 + 0.1 * (a - a.mean()) / (a.std() + 1e-8))
        
        return u


class HeatEquationDataset(BasePDEDataset):
    """
    Generate synthetic data for heat equation with variable diffusivity:
    ∂u/∂t = ∇·(a(x,y)∇u)
    
    We solve the steady-state version: ∇·(a∇u) = 0 with boundary conditions
    """
    
    def __init__(self, n_samples=1000, grid_size=64, seed=42):
        super().__init__(n_samples, grid_size, seed)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _generate_all_data(self):
        """Generate all data samples upfront"""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.data = []
        for idx in range(self.n_samples):
            sample_seed = self.seed + idx * 1000
            np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            
            # Generate diffusivity field a(x,y)
            a_log = self._generate_random_field(self.grid_size, power_decay=2.5)
            a = np.exp(a_log)  # Ensure positive
            a = a / a.max() * 5.0 + 0.1  # Scale to [0.1, 5.1]
            
            # Solve heat equation with boundary conditions
            # Simplified: use source term approach
            f = self._generate_random_field(self.grid_size, power_decay=1.0)
            
            u = self._solve_pde(a, f)
            
            self.data.append((torch.FloatTensor(a), torch.FloatTensor(u)))
    
    def _solve_pde(self, a, f):
        """Solve steady-state heat equation with source"""
        # Simplified Fourier-based solution
        a_fft = np.fft.fft2(a)
        f_fft = np.fft.fft2(f)
        
        kx = np.fft.fftfreq(self.grid_size)
        ky = np.fft.fftfreq(self.grid_size)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        K2[0, 0] = 1
        
        # Approximate: use average diffusivity
        a_avg = a.mean()
        u_fft = f_fft / (a_avg * K2 + 1e-6)
        u = np.real(np.fft.ifft2(u_fft))
        
        return u


class WaveEquationDataset(BasePDEDataset):
    """
    Generate synthetic data for wave equation with variable speed:
    ∂²u/∂t² = c²(x,y)∇²u
    
    Steady-state version or time-harmonic solution
    """
    
    def __init__(self, n_samples=1000, grid_size=64, seed=42, frequency=2.0):
        super().__init__(n_samples, grid_size, seed)
        self.frequency = frequency
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _generate_all_data(self):
        """Generate all data samples upfront"""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.data = []
        for idx in range(self.n_samples):
            sample_seed = self.seed + idx * 1000
            np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            
            # Generate wave speed field c(x,y)
            c = self._generate_random_field(self.grid_size, power_decay=2.0)
            c = np.exp(c)  # Ensure positive
            c = c / c.max() * 3.0 + 0.5  # Scale to [0.5, 3.5]
            
            # Generate source
            f = self._generate_random_field(self.grid_size, power_decay=1.5)
            
            # Solve wave equation (Helmholtz equation)
            u = self._solve_pde(c, f)
            
            self.data.append((torch.FloatTensor(c), torch.FloatTensor(u)))
    
    def _solve_pde(self, c, f):
        """Solve Helmholtz equation: -∇²u - (ω/c)²u = f"""
        omega = 2 * np.pi * self.frequency
        c_avg = c.mean()
        
        f_fft = np.fft.fft2(f)
        kx = np.fft.fftfreq(self.grid_size)
        ky = np.fft.fftfreq(self.grid_size)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        K2[0, 0] = 1
        
        # Helmholtz operator in Fourier space
        helmholtz = K2 - (omega / c_avg) ** 2
        helmholtz[0, 0] = 1
        
        u_fft = -f_fft / (helmholtz + 1e-6)
        u = np.real(np.fft.ifft2(u_fft))
        
        return u


# Dataset registry for easy access
DATASET_REGISTRY = {
    'poisson': PoissonDataset,
    'darcy': DarcyDataset,
    'heat': HeatEquationDataset,
    'wave': WaveEquationDataset,
}


def get_dataset(name: str, **kwargs) -> Dataset:
    """Get a dataset by name"""
    if name.lower() not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name.lower()](**kwargs)

