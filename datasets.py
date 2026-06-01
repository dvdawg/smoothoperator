"""Genuine PDE operator-learning datasets for (Condition-Aware) FNO benchmarks.

Every dataset defines an input field ``a`` and a target field ``u`` such that
``u`` is the output of a *real* solution operator applied to ``a``.  In
contrast to a forcing-driven surrogate, here the input fully determines the
output, so the learned map ``a -> u`` is a well-posed operator-learning
problem.

Conventions
-----------
* Domain is the periodic torus ``[0, 1]^2`` discretised on an ``N x N`` grid.
* Spatial derivatives use integer wavenumbers ``k`` (cycles per unit domain),
  so ``d/dx <-> i * 2*pi * k``.
* Inputs are returned channels-last with shape ``(N, N, C_in)`` (``C_in >= 1``)
  and targets with shape ``(N, N)`` (single output channel).  Multichannel
  inputs (``C_in > 1``) make the per-mode condition number non-trivial, which
  is required for the condition-aware machinery to be meaningful.

All solvers are vectorised over the sample axis with batched FFTs so that
regenerating thousands of samples per seed stays cheap.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# Spectral helpers (all batched over the leading sample axis)                 #
# --------------------------------------------------------------------------- #
def _wavenumbers(n: int):
    """Integer wavenumbers (cycles/domain) and squared magnitude on an N x N grid."""
    k = np.fft.fftfreq(n) * n  # integer cycles per domain
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    return kx, ky, k2


def _grf_batch(n_samples: int, n: int, alpha: float, rng: np.random.Generator):
    """Batch of zero-mean Gaussian random fields with spectral decay |k|^{-alpha}.

    Returns a real array of shape (n_samples, n, n), each field normalised to
    unit standard deviation.
    """
    _, _, k2 = _wavenumbers(n)
    kmag = np.sqrt(k2)
    amp = np.zeros_like(kmag)
    nz = kmag > 0
    amp[nz] = kmag[nz] ** (-alpha / 2.0)  # power ~ |k|^{-alpha}
    amp = amp[None, :, :]

    noise = rng.standard_normal((n_samples, n, n)) + 1j * rng.standard_normal((n_samples, n, n))
    field_ft = amp * noise
    field = np.fft.ifft2(field_ft, axes=(-2, -1)).real
    # normalise each sample to zero mean / unit std
    field -= field.mean(axis=(-2, -1), keepdims=True)
    std = field.std(axis=(-2, -1), keepdims=True)
    field /= std + 1e-8
    return field


# --------------------------------------------------------------------------- #
# Tensor packaging                                                            #
# --------------------------------------------------------------------------- #
def _to_dataset(a: np.ndarray, u: np.ndarray) -> TensorDataset:
    """Pack numpy arrays into a TensorDataset.

    ``a`` may be (S, N, N) for a single input channel or (S, N, N, C) for
    multichannel.  ``u`` is (S, N, N).  Single-channel inputs are promoted to a
    trailing channel axis so the model always receives channels-last input.
    """
    if a.ndim == 3:
        a = a[..., None]
    a_t = torch.from_numpy(np.ascontiguousarray(a)).float()
    u_t = torch.from_numpy(np.ascontiguousarray(u)).float()
    return TensorDataset(a_t, u_t)


# --------------------------------------------------------------------------- #
# Datasets                                                                    #
# --------------------------------------------------------------------------- #
def make_poisson(n_samples, grid_size, rng):
    """Elliptic, smoothing. -Delta u = f, periodic, zero-mean. Input f -> output u.

    Single channel. Spectral decay of u is two orders faster than f, so the
    output is strongly low-mode dominated (energy-truncation-friendly).
    """
    n = grid_size
    _, _, k2 = _wavenumbers(n)
    inv = np.zeros_like(k2)
    nz = k2 > 0
    inv[nz] = 1.0 / (TWO_PI ** 2 * k2[nz])

    f = _grf_batch(n_samples, n, alpha=2.0, rng=rng)
    f_ft = np.fft.fft2(f, axes=(-2, -1))
    u = np.fft.ifft2(f_ft * inv[None], axes=(-2, -1)).real
    return _to_dataset(f, u)


def make_heat(n_samples, grid_size, rng, nu=0.01, t_final=0.05):
    """Parabolic time evolution (heat semigroup). u(T) = exp(nu T Delta) u0.

    Single channel: input u0 -> output u(T). Strong high-frequency damping
    e^{-nu (2 pi)^2 |k|^2 T}, so retained-mode set should shrink (smoothing).
    """
    n = grid_size
    _, _, k2 = _wavenumbers(n)
    decay = np.exp(-nu * (TWO_PI ** 2) * k2 * t_final)

    u0 = _grf_batch(n_samples, n, alpha=1.0, rng=rng)  # rough IC, broad spectrum
    u0_ft = np.fft.fft2(u0, axes=(-2, -1))
    uT = np.fft.ifft2(u0_ft * decay[None], axes=(-2, -1)).real
    return _to_dataset(u0, uT)


def make_wave(n_samples, grid_size, rng, c=1.0, t_final=0.2):
    """Hyperbolic, energy-conserving. u_tt = c^2 Delta u, periodic.

    MULTICHANNEL (C_in = 2): input (u0, v0) -> output u(T), where
        u_hat(k,T) = cos(omega_k T) u0_hat + sin(omega_k T)/omega_k v0_hat,
        omega_k = c * 2 pi |k|.
    Energy is spread across all frequencies (no damping), so the operator
    genuinely needs high-wavenumber modes; the two input channels give a
    non-trivial per-mode condition number.
    """
    n = grid_size
    _, _, k2 = _wavenumbers(n)
    kmag = np.sqrt(k2)
    omega = c * TWO_PI * kmag
    cos_t = np.cos(omega * t_final)
    sinc_t = np.where(omega > 0, np.sin(omega * t_final) / np.where(omega > 0, omega, 1.0), t_final)

    u0 = _grf_batch(n_samples, n, alpha=1.5, rng=rng)
    v0 = _grf_batch(n_samples, n, alpha=1.5, rng=rng)
    u0_ft = np.fft.fft2(u0, axes=(-2, -1))
    v0_ft = np.fft.fft2(v0, axes=(-2, -1))
    uT_ft = cos_t[None] * u0_ft + sinc_t[None] * v0_ft
    uT = np.fft.ifft2(uT_ft, axes=(-2, -1)).real

    a = np.stack([u0, v0], axis=-1)  # (S, N, N, 2)
    return _to_dataset(a, uT)


def make_advection_diffusion(n_samples, grid_size, rng, bx=1.0, by=0.6, nu=0.01, t_final=0.1):
    """Hyperbolic-parabolic transport. u_t + b.grad u = nu Delta u, periodic.

    Single channel: input u0 -> output u(T). The transport part is an
    anisotropic phase shift (no damping); the diffusion part damps high modes.
    The anisotropy of b makes the directional spectral structure non-trivial,
    a good test for the 2D condition-aware geometry.
    """
    n = grid_size
    kx, ky, k2 = _wavenumbers(n)
    symbol = np.exp(-(1j * TWO_PI * (bx * kx + by * ky) + nu * (TWO_PI ** 2) * k2) * t_final)

    u0 = _grf_batch(n_samples, n, alpha=1.5, rng=rng)
    u0_ft = np.fft.fft2(u0, axes=(-2, -1))
    uT = np.fft.ifft2(u0_ft * symbol[None], axes=(-2, -1)).real
    return _to_dataset(u0, uT)


def _solve_darcy_batch(a, f, rng=None, n_iter=400, tol=1e-6):
    """Variable-coefficient elliptic solve  -div(a grad u) = f  on the periodic torus.

    Standard cell-centred finite-volume discretisation with harmonic-mean face
    conductivities -- a symmetric positive-(semi)definite operator with no
    aliasing.  Solved by matrix-free preconditioned conjugate gradient, with the
    constant-coefficient finite-difference Laplacian (diagonalised by the DFT)
    as preconditioner.  Fully batched over the sample axis; both ``u`` and ``f``
    are taken zero-mean (the periodic problem is defined up to a constant).
    """
    s, n, _ = a.shape
    dx2 = (1.0 / n) ** 2

    # harmonic-mean face conductivities (periodic neighbours via roll)
    def harm(x, y):
        return 2.0 * x * y / (x + y + 1e-30)

    a_xp = harm(a, np.roll(a, -1, axis=-2))  # face between (i,j) and (i+1,j)
    a_xm = harm(a, np.roll(a, 1, axis=-2))   # face between (i,j) and (i-1,j)
    a_yp = harm(a, np.roll(a, -1, axis=-1))
    a_ym = harm(a, np.roll(a, 1, axis=-1))

    def apply_A(u):
        flux = (
            a_xp * (u - np.roll(u, -1, axis=-2))
            + a_xm * (u - np.roll(u, 1, axis=-2))
            + a_yp * (u - np.roll(u, -1, axis=-1))
            + a_ym * (u - np.roll(u, 1, axis=-1))
        )
        return flux / dx2

    # FFT preconditioner: constant-coefficient FD Laplacian eigenvalues
    a_bar = a.mean(axis=(-2, -1), keepdims=True)  # (s,1,1)
    k = np.arange(n)
    lam = (2.0 - 2.0 * np.cos(TWO_PI * k / n)) / dx2
    lam2d = lam[:, None] + lam[None, :]  # (n,n)
    inv_eig = np.zeros_like(lam2d)
    inv_eig[lam2d > 0] = 1.0 / lam2d[lam2d > 0]
    inv_eig = inv_eig[None]

    def _zero_mean(v):
        return v - v.mean(axis=(-2, -1), keepdims=True)

    def precond(r):
        rft = np.fft.fft2(r, axes=(-2, -1))
        return np.fft.ifft2(rft * inv_eig, axes=(-2, -1)).real / a_bar

    def dot(p, q):  # per-sample inner product -> (s,1,1)
        return np.sum(p * q, axis=(-2, -1), keepdims=True)

    f = _zero_mean(f)
    fnorm = np.sqrt(dot(f, f)) + 1e-30

    x = np.zeros_like(f)
    r = f.copy()
    z = _zero_mean(precond(r))
    p = z.copy()
    rz = dot(r, z)
    for _ in range(n_iter):
        Ap = _zero_mean(apply_A(p))
        alpha = rz / (dot(p, Ap) + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        if np.max(np.sqrt(dot(r, r)) / fnorm) < tol:
            break
        z = _zero_mean(precond(r))
        rz_new = dot(r, z)
        beta = rz_new / (rz + 1e-30)
        p = z + beta * p
        rz = rz_new

    return _zero_mean(x)


def make_darcy(n_samples, grid_size, rng, contrast=4.0):
    """Variable-coefficient elliptic (Darcy flow). -div(a grad u) = f, fixed f.

    Single channel: input log-permeability-derived a(x) -> output pressure u.
    The variable coefficient genuinely *couples* Fourier modes (unlike a
    convolution), so the per-mode diagonal regression is only approximate --
    the realistic regime an FNO is meant to handle.
    """
    n = grid_size
    log_a = _grf_batch(n_samples, n, alpha=2.0, rng=rng)
    a = np.exp(0.5 * np.log(contrast) * log_a)  # contrast-controlled permeability
    # fixed deterministic forcing shared across samples
    fx = np.sin(TWO_PI * np.linspace(0, 1, n, endpoint=False))
    f = np.broadcast_to(fx[None, :, None] + 0.0 * fx[None, None, :], (n_samples, n, n)).copy()
    f = f + np.cos(TWO_PI * np.linspace(0, 1, n, endpoint=False))[None, None, :]
    u = _solve_darcy_batch(a, f, rng)
    return _to_dataset(a, u)


def make_darcy_multi(n_samples, grid_size, rng, contrast=4.0):
    """Variable-coefficient Darcy with BOTH coefficient and forcing varying.

    MULTICHANNEL (C_in = 2): input (a, f) -> output u solving -div(a grad u)=f.
    Both the permeability and the source vary per sample, so the two input
    channels carry independent information and the per-mode condition number is
    genuinely non-trivial -- the canonical multichannel operator-learning task.
    """
    n = grid_size
    log_a = _grf_batch(n_samples, n, alpha=2.0, rng=rng)
    a = np.exp(0.5 * np.log(contrast) * log_a)
    f = _grf_batch(n_samples, n, alpha=1.5, rng=rng)
    u = _solve_darcy_batch(a, f, rng)
    inp = np.stack([a, f], axis=-1)  # (S, N, N, 2)
    return _to_dataset(inp, u)


def make_heat_sensor(n_samples, grid_size, rng, noise=0.15, t_final=0.05, nu=0.01):
    """Heat evolution observed through two correlated noisy sensors.

    MULTICHANNEL (C_in = 2): input = (u0 + n1, u0 + n2) -> output u(T), the heat
    semigroup applied to the *clean* u0.  Because both channels observe the same
    field u0, at signal-dominated (low) wavenumbers the two columns of the
    per-mode data matrix are nearly collinear -> large condition number; at
    noise-dominated (high) wavenumbers they are independent -> condition number
    near 1.  This is a controlled setting in which the energy and conditioning
    terms of the utility score genuinely disagree, so condition-aware selection
    differs from pure energy truncation.
    """
    n = grid_size
    _, _, k2 = _wavenumbers(n)
    decay = np.exp(-nu * (TWO_PI ** 2) * k2 * t_final)

    u0 = _grf_batch(n_samples, n, alpha=1.0, rng=rng)
    n1 = noise * rng.standard_normal((n_samples, n, n))
    n2 = noise * rng.standard_normal((n_samples, n, n))
    u0_ft = np.fft.fft2(u0, axes=(-2, -1))
    uT = np.fft.ifft2(u0_ft * decay[None], axes=(-2, -1)).real

    a = np.stack([u0 + n1, u0 + n2], axis=-1)  # (S, N, N, 2)
    return _to_dataset(a, uT)


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #
DATASET_BUILDERS = {
    "poisson": make_poisson,
    "heat": make_heat,
    "wave": make_wave,                  # multichannel (2)
    "advection": make_advection_diffusion,
    "darcy": make_darcy,
    "darcy_multi": make_darcy_multi,    # multichannel (2)
    "heat_sensor": make_heat_sensor,    # multichannel (2), ill-conditioned modes
}

# input-channel count per dataset (used by the model factory)
DATASET_IN_CHANNELS = {
    "poisson": 1,
    "heat": 1,
    "wave": 2,
    "advection": 1,
    "darcy": 1,
    "darcy_multi": 2,
    "heat_sensor": 2,
}

DATASET_REGISTRY = DATASET_BUILDERS  # backwards-compatible alias


def dataset_in_channels(name: str) -> int:
    return DATASET_IN_CHANNELS[name.lower()]


def get_dataset(name: str, n_samples: int = 1000, grid_size: int = 64, seed: int = 42, **kwargs):
    """Build a dataset by name. Returns a torch Dataset of (input, target) pairs."""
    key = name.lower()
    if key not in DATASET_BUILDERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_BUILDERS)}")
    rng = np.random.default_rng(seed)
    return DATASET_BUILDERS[key](n_samples, grid_size, rng, **kwargs)


def make_pde_dataloaders(
    name: str,
    n_train: int,
    n_test: int,
    grid_size: int = 64,
    batch_size: int = 20,
    seed: int = 42,
):
    train_ds = get_dataset(name, n_samples=n_train, grid_size=grid_size, seed=seed)
    test_ds = get_dataset(name, n_samples=n_test, grid_size=grid_size, seed=seed + 1000)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
