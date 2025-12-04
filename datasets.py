"""
Dataset generators for testing Condition-Aware FNO.

Includes multiple PDE datasets:
- Poisson equation
- Darcy-like flow
- Heat equation
- Wave equation
"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BasePDEDataset(Dataset):
    def __init__(self, n_samples: int = 1000, grid_size: int = 64, seed: int = 42):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.seed = seed
        self.data = []
        self._generate_all_data()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        return self.data[idx]

    def _set_sample_seed(self, idx: int) -> None:
        s = self.seed + idx
        np.random.seed(s)
        torch.manual_seed(s)

    def _generate_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _generate_all_data(self) -> None:
        self.data = []
        for i in range(self.n_samples):
            self._set_sample_seed(i)
            a, u = self._generate_sample(i)
            self.data.append((a, u))


class PoissonDataset(BasePDEDataset):
    """
    Synthetic Poisson dataset solving -Δu = a on a periodic square domain.
    """

    def __init__(self, n_samples: int = 1000, grid_size: int = 64, seed: int = 42):
        super().__init__(n_samples=n_samples, grid_size=grid_size, seed=seed)

    def _generate_random_field(self, size: int) -> np.ndarray:
        kx = np.fft.fftfreq(size)
        ky = np.fft.fftfreq(size)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX ** 2 + KY ** 2

        power = 1.0 / (1.0 + K2)
        power[0, 0] = 0.0

        phase = np.random.randn(size, size) + 1j * np.random.randn(size, size)
        field_fft = np.sqrt(power) * phase
        field = np.real(np.fft.ifft2(field_fft))

        field = (field - field.mean()) / (field.std() + 1e-8)
        return field

    def _solve_poisson(self, a: np.ndarray) -> np.ndarray:
        a_fft = np.fft.fft2(a)
        kx = np.fft.fftfreq(a.shape[0])
        ky = np.fft.fftfreq(a.shape[1])
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX ** 2 + KY ** 2
        K2[0, 0] = 1.0
        u_fft = a_fft / K2
        u = np.real(np.fft.ifft2(u_fft))
        return u

    def _generate_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self._generate_random_field(self.grid_size)
        u = self._solve_poisson(a)
        a_t = torch.from_numpy(a).float()
        u_t = torch.from_numpy(u).float()
        return a_t, u_t


class DarcyDataset(BasePDEDataset):
    """
    Simplified Darcy-like dataset.

    We generate a random log-permeability field a(x, y) and construct a target
    u(x, y) by solving a Poisson problem with a forcing derived from a.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        grid_size: int = 64,
        seed: int = 42,
        log_amplitude: float = 0.5,
    ):
        self.log_amplitude = log_amplitude
        super().__init__(n_samples=n_samples, grid_size=grid_size, seed=seed)

    def _generate_log_perm(self, size: int) -> np.ndarray:
        kx = np.fft.fftfreq(size)
        ky = np.fft.fftfreq(size)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX ** 2 + KY ** 2
        power = 1.0 / (1.0 + K2)
        power[0, 0] = 0.0
        phase = np.random.randn(size, size) + 1j * np.random.randn(size, size)
        field_fft = np.sqrt(power) * phase
        base = np.real(np.fft.ifft2(field_fft))
        base = (base - base.mean()) / (base.std() + 1e-8)
        log_k = self.log_amplitude * base
        return log_k

    def _solve_poisson(self, f: np.ndarray) -> np.ndarray:
        f_fft = np.fft.fft2(f)
        kx = np.fft.fftfreq(f.shape[0])
        ky = np.fft.fftfreq(f.shape[1])
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX ** 2 + KY ** 2
        K2[0, 0] = 1.0
        u_fft = f_fft / K2
        u = np.real(np.fft.ifft2(u_fft))
        return u

    def _laplacian(self, a: np.ndarray) -> np.ndarray:
        kx = np.fft.fftfreq(a.shape[0])
        ky = np.fft.fftfreq(a.shape[1])
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX ** 2 + KY ** 2
        a_fft = np.fft.fft2(a)
        lap_fft = -K2 * a_fft
        lap = np.real(np.fft.ifft2(lap_fft))
        return lap

    def _generate_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        log_k = self._generate_log_perm(self.grid_size)
        a = np.exp(log_k)
        lap_a = self._laplacian(a)
        u = self._solve_poisson(lap_a)

        a_t = torch.from_numpy(a).float()
        u_t = torch.from_numpy(u).float()
        return a_t, u_t


class HeatEquationDataset(BasePDEDataset):
    """
    Heat equation dataset: u_t = ν Δu, with periodic boundary conditions.

    Input a(x, y) is the initial condition u(x, y, 0).
    Output u(x, y) is the solution at time T.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        grid_size: int = 64,
        seed: int = 42,
        diffusivity: float = 0.01,
        final_time: float = 1.0,
    ):
        self.diffusivity = diffusivity
        self.final_time = final_time
        super().__init__(n_samples=n_samples, grid_size=grid_size, seed=seed)

    def _generate_initial_condition(self, size: int) -> np.ndarray:
        kx = np.fft.fftfreq(size)
        ky = np.fft.fftfreq(size)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX ** 2 + KY ** 2
        power = 1.0 / (1.0 + K2)
        power[0, 0] = 0.0
        phase = np.random.randn(size, size) + 1j * np.random.randn(size, size)
        field_fft = np.sqrt(power) * phase
        field = np.real(np.fft.ifft2(field_fft))
        field = (field - field.mean()) / (field.std() + 1e-8)
        return field

    def _solve_heat(self, u0: np.ndarray) -> np.ndarray:
        kx = np.fft.fftfreq(u0.shape[0])
        ky = np.fft.fftfreq(u0.shape[1])
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX ** 2 + KY ** 2
        u0_fft = np.fft.fft2(u0)
        propagator = np.exp(-4.0 * (np.pi ** 2) * self.diffusivity * K2 * self.final_time)
        uT_fft = u0_fft * propagator
        uT = np.real(np.fft.ifft2(uT_fft))
        return uT

    def _generate_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self._generate_initial_condition(self.grid_size)
        u = self._solve_heat(a)
        a_t = torch.from_numpy(a).float()
        u_t = torch.from_numpy(u).float()
        return a_t, u_t


class WaveEquationDataset(BasePDEDataset):
    """
    Wave equation dataset: u_tt = c^2 Δu, with initial velocity 0.

    Input a(x, y) is the initial displacement u(x, y, 0).
    Output u(x, y) is the displacement at time T.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        grid_size: int = 64,
        seed: int = 42,
        wave_speed: float = 1.0,
        final_time: float = 1.0,
    ):
        self.wave_speed = wave_speed
        self.final_time = final_time
        super().__init__(n_samples=n_samples, grid_size=grid_size, seed=seed)

    def _generate_initial_displacement(self, size: int) -> np.ndarray:
        kx = np.fft.fftfreq(size)
        ky = np.fft.fftfreq(size)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX ** 2 + KY ** 2
        power = 1.0 / (1.0 + K2)
        power[0, 0] = 0.0
        phase = np.random.randn(size, size) + 1j * np.random.randn(size, size)
        field_fft = np.sqrt(power) * phase
        field = np.real(np.fft.ifft2(field_fft))
        field = (field - field.mean()) / (field.std() + 1e-8)
        return field

    def _solve_wave(self, u0: np.ndarray) -> np.ndarray:
        kx = np.fft.fftfreq(u0.shape[0])
        ky = np.fft.fftfreq(u0.shape[1])
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX ** 2 + KY ** 2)

        u0_fft = np.fft.fft2(u0)
        omega = 2.0 * np.pi * self.wave_speed * K
        cos_term = np.cos(omega * self.final_time)
        uT_fft = u0_fft * cos_term
        uT = np.real(np.fft.ifft2(uT_fft))
        return uT

    def _generate_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self._generate_initial_displacement(self.grid_size)
        u = self._solve_wave(a)
        a_t = torch.from_numpy(a).float()
        u_t = torch.from_numpy(u).float()
        return a_t, u_t


DATASET_REGISTRY = {
    "poisson": PoissonDataset,
    "darcy": DarcyDataset,
    "heat": HeatEquationDataset,
    "wave": WaveEquationDataset,
}


def get_dataset(name: str, **kwargs) -> Dataset:
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](**kwargs)
