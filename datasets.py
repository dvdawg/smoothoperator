import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

class BasePDEDataset(Dataset):    
    def __init__(self, n_samples=1000, grid_size=64, seed=42):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.seed = seed
                                                    
        self._generate_all_data()
    
    def _generate_all_data(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.data = []
        for idx in range(self.n_samples):
                                                                            
            sample_seed = self.seed + idx * 1000
            np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            
            a = self._generate_random_field(self.grid_size)
            u = self._solve_pde(a)
            self.data.append((torch.FloatTensor(a), torch.FloatTensor(u)))
    
    def _solve_pde(self, a):
        raise NotImplementedError("Subclasses must implement _solve_pde")
    
    def _generate_random_field(self, size, power_decay=1.0):                                 
        kx = np.fft.fftfreq(size)
        ky = np.fft.fftfreq(size)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)
        
                                               
        power = 1.0 / (1.0 + K**power_decay)
        power[0, 0] = 0                       
        
                               
        phase = np.random.randn(size, size) + 1j * np.random.randn(size, size)
        a_fft = np.sqrt(power) * phase
        a = np.real(np.fft.ifft2(a_fft))
        
                   
        a = (a - a.mean()) / (a.std() + 1e-8)
        return a

class PoissonDataset(BasePDEDataset):    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _solve_pde(self, a):
        a_fft = np.fft.fft2(a)
        kx = np.fft.fftfreq(a.shape[0])
        ky = np.fft.fftfreq(a.shape[1])
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        K2[0, 0] = 1                          
        
        u_fft = a_fft / K2
        u = np.real(np.fft.ifft2(u_fft))
        return u

class DarcyDataset(BasePDEDataset):

    def __init__(self, n_samples=1000, grid_size=64, seed=42, permeability_range=(0.1, 10.0)):
        self.permeability_range = permeability_range
        super().__init__(n_samples, grid_size, seed)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _generate_all_data(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.data = []
        for idx in range(self.n_samples):
            sample_seed = self.seed + idx * 1000
            np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            
                                                
            a_log = self._generate_random_field(self.grid_size, power_decay=2.0)
                                       
            a_min, a_max = self.permeability_range
            a = np.exp(a_log * np.log(a_max / a_min) / 2 + np.log((a_min * a_max) ** 0.5))
            
                                    
            f = self._generate_random_field(self.grid_size, power_decay=1.5)
            f = (f - f.min()) / (f.max() - f.min() + 1e-8) * 2 - 1                        
            
                                                
            u = self._solve_pde(a, f)
            
            self.data.append((torch.FloatTensor(a), torch.FloatTensor(u)))
    
    def _solve_pde(self, a, f):                                                                 
        dx = 1.0 / self.grid_size                                                                             
        a_avg = a.mean()
        f_fft = np.fft.fft2(f)
        kx = np.fft.fftfreq(self.grid_size)
        ky = np.fft.fftfreq(self.grid_size)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        K2[0, 0] = 1
        
                              
        u_fft = f_fft / (a_avg * K2 + 1e-6)
        u = np.real(np.fft.ifft2(u_fft))
        
                                                                       
                                                                  
        u = u * (1 + 0.1 * (a - a.mean()) / (a.std() + 1e-8))
        
        return u

class HeatEquationDataset(BasePDEDataset):
    
    def __init__(self, n_samples=1000, grid_size=64, seed=42):
        super().__init__(n_samples, grid_size, seed)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _generate_all_data(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.data = []
        for idx in range(self.n_samples):
            sample_seed = self.seed + idx * 1000
            np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)
                                               
            a_log = self._generate_random_field(self.grid_size, power_decay=2.5)
            a = np.exp(a_log)                   
            a = a / a.max() * 5.0 + 0.1                       
            f = self._generate_random_field(self.grid_size, power_decay=1.0)
            
            u = self._solve_pde(a, f)
            
            self.data.append((torch.FloatTensor(a), torch.FloatTensor(u)))
    
    def _solve_pde(self, a, f):                                           
        a_fft = np.fft.fft2(a)
        f_fft = np.fft.fft2(f)
        
        kx = np.fft.fftfreq(self.grid_size)
        ky = np.fft.fftfreq(self.grid_size)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        K2[0, 0] = 1
        
                                              
        a_avg = a.mean()
        u_fft = f_fft / (a_avg * K2 + 1e-6)
        u = np.real(np.fft.ifft2(u_fft))
        
        return u

class WaveEquationDataset(BasePDEDataset):
    def __init__(self, n_samples=1000, grid_size=64, seed=42, frequency=2.0):
        self.frequency = frequency
        super().__init__(n_samples, grid_size, seed)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _generate_all_data(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.data = []
        for idx in range(self.n_samples):
            sample_seed = self.seed + idx * 1000
            np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            
                                              
            c = self._generate_random_field(self.grid_size, power_decay=2.0)
            c = np.exp(c)                   
            c = c / c.max() * 3.0 + 0.5                       
            
                             
            f = self._generate_random_field(self.grid_size, power_decay=1.5)
            
                                                      
            u = self._solve_pde(c, f)
            
            self.data.append((torch.FloatTensor(c), torch.FloatTensor(u)))
    
    def _solve_pde(self, c, f):
        omega = 2 * np.pi * self.frequency
        c_avg = c.mean()
        
        f_fft = np.fft.fft2(f)
        kx = np.fft.fftfreq(self.grid_size)
        ky = np.fft.fftfreq(self.grid_size)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        K2[0, 0] = 1
        
                                             
        helmholtz = K2 - (omega / c_avg) ** 2
        helmholtz[0, 0] = 1
        
        u_fft = -f_fft / (helmholtz + 1e-6)
        u = np.real(np.fft.ifft2(u_fft))
        
        return u

                                  
DATASET_REGISTRY = {
    'poisson': PoissonDataset,
    'darcy': DarcyDataset,
    'heat': HeatEquationDataset,
    'wave': WaveEquationDataset,
}

def get_dataset(name: str, **kwargs) -> Dataset:
    if name.lower() not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name.lower()](**kwargs)

def make_pde_dataloaders(
    name: str,
    n_train: int,
    n_test: int,
    grid_size: int = 64,
    batch_size: int = 20,
    seed: int = 42,
):

    train_dataset = get_dataset(name, n_samples=n_train, grid_size=grid_size, seed=seed)
    test_dataset = get_dataset(name, n_samples=n_test, grid_size=grid_size, seed=seed + 1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
