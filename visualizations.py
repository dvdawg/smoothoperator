import sys, os
sys.path.insert(0, '/mnt/user-data/uploads')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from datasets import PoissonDataset, HeatEquationDataset, WaveEquationDataset
from condition_aware_fno import (
    FNO2d, ConditionAwareFNO2d,
    compute_adaptive_mask, train_epoch, evaluate
)

os.makedirs('poster_figures', exist_ok=True)

                                                                               
C_STD  = '#E05A2B'                                 
C_CA   = '#2E86AB'                                        
C_GRID = '#CCCCCC'
BG     = '#F8F8F5'

DATASETS = {
    'poisson': {'cls': PoissonDataset,      'label': 'Poisson (Elliptic)',    'color': '#9B59B6'},
    'heat':    {'cls': HeatEquationDataset, 'label': 'Heat (Parabolic)',      'color': '#E74C3C'},
    'wave':    {'cls': WaveEquationDataset, 'label': 'Wave (Hyperbolic)',     'color': '#27AE60'},
}

N_VIS   = 200                                  
GRID    = 64
MODES   = 12
DEVICE  = torch.device('cpu')

plt.rcParams.update({
    'font.family':      'DejaVu Serif',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.facecolor':   BG,
    'figure.facecolor': BG,
    'axes.grid':        True,
    'grid.color':       C_GRID,
    'grid.linewidth':   0.6,
    'grid.alpha':       0.6,
})

                                                                                 
         
                                                                                 

def load_dataset(name, n=N_VIS):
    ds = DATASETS[name]['cls'](n_samples=n, grid_size=GRID, seed=42)
    a_list, u_list = zip(*[ds[i] for i in range(n)])
    a = torch.stack(a_list)              
    u = torch.stack(u_list)
    return a, u

def compute_mode_stats(a, u, modes=MODES):
    a4 = a.unsqueeze(1)
    u4 = u.unsqueeze(1)
    a_fft = torch.fft.rfft2(a4)                      
    u_fft = torch.fft.rfft2(u4)
    _, _, H, W_r = a_fft.shape
    m2 = min(modes, W_r)
    m1 = min(modes, H)

    energy  = np.zeros((m1, m2))
    cond    = np.zeros((m1, m2))
    scores  = np.zeros((m1, m2))

    for i in range(m1):
        for j in range(m2):
            Xk = a_fft[:, :, i, j]                           
            Yk = u_fft[:, :, i, j]
            e = torch.sum(torch.abs(Yk)**2).real.item()
            energy[i, j] = e
            try:
                sv = torch.linalg.svdvals(Xk)
                kappa = (sv[0] / sv[-1]).item() if sv[-1] > 1e-9 else 1e9
            except Exception:
                kappa = 1e9
            cond[i, j]   = min(kappa, 1e6)
            scores[i, j] = e / (kappa + 1e-8)

    return energy, cond, scores

def compute_psd_1d(u):
    u_fft = np.fft.fft2(u.numpy())
    power = np.abs(u_fft)**2
    H, W  = power.shape
    kx = np.fft.fftfreq(W, d=1.0/W)
    ky = np.fft.fftfreq(H, d=1.0/H)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2).flatten()
    P = power.flatten()
    k_int  = np.round(K).astype(int)
    k_max  = int(K.max())
    psd    = np.zeros(k_max + 1)
    counts = np.zeros(k_max + 1)
    for ki, pi in zip(k_int, P):
        if 0 <= ki <= k_max:
            psd[ki] += pi
            counts[ki] += 1
    counts[counts == 0] = 1
    return np.arange(k_max + 1), psd / counts

def save(fig, name):
    path = f'poster_figures/{name}.png'
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  Saved → {path}')

                                                                                 
                                  
                                                                                 
print('Fig 1: solution field snapshots …')

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
fig.suptitle('PDE Solution Fields', fontsize=15, fontweight='bold', y=1.02)

for ax, (name, meta) in zip(axes, DATASETS.items()):
    a, u = load_dataset(name, n=5)
    sample = u[0].numpy()
    im = ax.imshow(sample, cmap='RdBu_r', interpolation='bilinear')
    ax.set_title(meta['label'], fontsize=12, color=meta['color'], fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)

plt.tight_layout()
save(fig, '1_solution_snapshots')

                                                                                 
                                          
                                                                                 
print('Fig 2: power spectral density …')

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title('Power Spectral Density by Dataset', fontsize=13, fontweight='bold')

for name, meta in DATASETS.items():
    _, u = load_dataset(name, n=50)
    psds = [compute_psd_1d(u[i]) for i in range(min(20, len(u)))]
    k_ref = psds[0][0]
    psd_mean = np.mean([p for _, p in psds], axis=0)
    ax.loglog(k_ref[1:], psd_mean[1:], color=meta['color'],
              linewidth=2.5, label=meta['label'])

                        
k_ref_line = np.array([1, 30])
for slope, ls, label in [(-2, '--', r'$k^{-2}$'), (-4, ':', r'$k^{-4}$')]:
    ax.loglog(k_ref_line, 1e6 * k_ref_line**slope, color='gray',
              linestyle=ls, linewidth=1.2, alpha=0.7, label=label)

ax.set_xlabel('Wavenumber $k$', fontsize=11)
ax.set_ylabel('Power $|\\hat{u}(k)|^2$', fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(1, GRID // 2)
plt.tight_layout()
save(fig, '2_power_spectral_density')

                                                                                 
                                                                               
                                                                                 
print('Fig 3: per-mode heatmaps …')

dataset_names = list(DATASETS.keys())
row_labels = ['Energy $\\|Y_k\\|^2$', 'Condition $\\kappa_k$', 'Score $E_k/\\kappa_k$']

fig = plt.figure(figsize=(14, 10))
fig.suptitle('Per-Mode Spectral Statistics (modes 0–11)', fontsize=14, fontweight='bold', y=1.01)
gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

for col, name in enumerate(dataset_names):
    a, u = load_dataset(name)
    energy, cond, scores = compute_mode_stats(a, u)

    for row, (data, title, cmap) in enumerate(zip(
        [energy, cond, scores],
        row_labels,
        ['YlOrRd', 'PuBu', 'YlGn']
    )):
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(data, cmap=cmap, interpolation='nearest', aspect='auto')
        if row == 0:
            ax.set_title(DATASETS[name]['label'], fontsize=11,
                         color=DATASETS[name]['color'], fontweight='bold')
        if col == 0:
            ax.set_ylabel(title, fontsize=9)
        ax.set_xlabel('$k_2$', fontsize=8)
        if col == 0:
            ax.set_yticks(range(0, MODES, 3))
        else:
            ax.set_yticks([])
        ax.set_xticks(range(0, MODES, 3))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='6%', pad=0.04)
        fig.colorbar(im, cax=cax)

save(fig, '3_mode_heatmaps')

                                                                                 
                                                          
                                                                                 
print('Fig 4: score heatmap + mask overlay …')

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle('Mode Selection: Score Map + Retained Modes (η = 0.95)',
             fontsize=13, fontweight='bold')

for ax, (name, meta) in zip(axes, DATASETS.items()):
    a, u = load_dataset(name)
    energy, cond, scores = compute_mode_stats(a, u)
    low_mask, _ = compute_adaptive_mask(a, u, MODES, MODES, energy_fraction=0.95)
    mask_np = low_mask.numpy()

    im = ax.imshow(scores, cmap='magma', interpolation='nearest', aspect='auto')
                                              
    for i in range(mask_np.shape[0]):
        for j in range(mask_np.shape[1]):
            if mask_np[i, j]:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                      linewidth=1.5, edgecolor='white',
                                      facecolor='none')
                ax.add_patch(rect)

    n_retained = mask_np.sum()
    n_total    = mask_np.size
    ax.set_title(f'{meta["label"]}\n({n_retained}/{n_total} modes retained)',
                 fontsize=10, color=meta['color'], fontweight='bold')
    ax.set_xlabel('$k_2$', fontsize=9)
    ax.set_ylabel('$k_1$', fontsize=9)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, label='Score')

                  
    ax.add_patch(mpatches.Patch(facecolor='none', edgecolor='white',
                                linewidth=1.5, label='Retained'))
    ax.legend(fontsize=8, loc='lower right', framealpha=0.4)

plt.tight_layout()
save(fig, '4_score_mask_overlay')

                                                                                 
                                
                                                                                 
print('Fig 5: loss landscape cartoon …')

def elliptical_loss(w1, w2, sigma1_sq, sigma2_sq, lam=0.0):
    return (sigma1_sq + lam) * w1**2 + (sigma2_sq + lam) * w2**2

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
fig.suptitle('Loss Landscape: Effect of Spectral Conditioning',
             fontsize=13, fontweight='bold')

w = np.linspace(-2, 2, 400)
W1, W2 = np.meshgrid(w, w)

configs = [
    dict(sigma1_sq=10.0, sigma2_sq=0.02, lam=0.0,
         title='Standard FNO\n(ill-conditioned, elongated)',
         color=C_STD, arrow_color=C_STD),
    dict(sigma1_sq=10.0, sigma2_sq=0.02, lam=1.5,
         title='Condition-Aware FNO\n(regularized, isotropic)',
         color=C_CA, arrow_color=C_CA),
]

for ax, cfg in zip(axes, configs):
    Z = elliptical_loss(W1, W2, cfg['sigma1_sq'], cfg['sigma2_sq'], cfg['lam'])
    levels = np.logspace(np.log10(Z.min() + 0.01), np.log10(Z.max()), 14)
    cs  = ax.contourf(W1, W2, Z, levels=levels, cmap='coolwarm', alpha=0.85)
    ax.contour(W1, W2, Z, levels=levels, colors='white', linewidths=0.4, alpha=0.5)

                                                                 
    pt = np.array([1.5, 1.5])
    path = [pt.copy()]
    lr = 0.05
    for _ in range(80):
        grad = np.array([
            2 * (cfg['sigma1_sq'] + cfg['lam']) * pt[0],
            2 * (cfg['sigma2_sq'] + cfg['lam']) * pt[1],
        ])
        pt = pt - lr * grad
        path.append(pt.copy())
        if np.linalg.norm(pt) < 0.05:
            break
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], color=cfg['arrow_color'],
            linewidth=2.5, zorder=5, label='GD path')
    ax.plot(path[0, 0], path[0, 1], 'o', color='white', markersize=7, zorder=6)
    ax.plot(0, 0, '*', color='gold', markersize=12, zorder=7, label='Minimum')

    kappa = (cfg['sigma1_sq'] + cfg['lam']) / (cfg['sigma2_sq'] + cfg['lam'])
    ax.set_title(f"{cfg['title']}\n$\\kappa = {kappa:.1f}$", fontsize=11,
                 color=cfg['color'], fontweight='bold')
    ax.set_xlabel('$w_1$', fontsize=10)
    ax.set_ylabel('$w_2$', fontsize=10)
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.legend(fontsize=9, loc='upper right')
    fig.colorbar(cs, ax=ax, label='Loss', shrink=0.85)

plt.tight_layout()
save(fig, '5_loss_landscape')

                                                                                 
                                      
                                                                                 
print('Fig 6: convergence rate bound …')

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title('Gradient Descent Convergence Rate Bound $(\\kappa-1)/(\\kappa+1)$',
             fontsize=12, fontweight='bold')

kappa_vals = np.logspace(0.01, 4, 500)
rate       = (kappa_vals - 1) / (kappa_vals + 1)
ax.semilogx(kappa_vals, rate, color='#2C3E50', linewidth=2.5)
ax.fill_between(kappa_vals, rate, 0, alpha=0.08, color='#2C3E50')

                                                               
markers = []
for name, meta in DATASETS.items():
    a, u = load_dataset(name, n=50)
    _, cond, _ = compute_mode_stats(a, u)
                                                     
    corner = cond[MODES//2:, MODES//2:]
    kappa_std = float(np.median(corner[corner < 1e5]))
    kappa_reg = kappa_std / (1 + 1.5 / (kappa_std + 1))                          
    rate_std = (kappa_std - 1) / (kappa_std + 1)
    rate_reg = (kappa_reg - 1) / (kappa_reg + 1)
    ax.scatter([kappa_std], [rate_std], color=meta['color'], s=100, zorder=5,
               marker='X', label=f"{meta['label']} – Standard")
    ax.scatter([kappa_reg],  [rate_reg],  color=meta['color'], s=100, zorder=5,
               marker='o', label=f"{meta['label']} – Cond.-Aware")
    ax.annotate('', xy=(kappa_reg, rate_reg), xytext=(kappa_std, rate_std),
                arrowprops=dict(arrowstyle='->', color=meta['color'],
                                lw=1.6, connectionstyle='arc3,rad=0.25'))

ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax.set_xlabel('Condition Number $\\kappa$', fontsize=11)
ax.set_ylabel('Convergence Rate Bound', fontsize=11)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, ncol=2, loc='lower right')
plt.tight_layout()
save(fig, '6_convergence_rate_bound')

                                                                                 
                                                                 
                                                    
                                                                                 
print('Fig 7: per-mode prediction error (training quick models) …')

from torch.utils.data import DataLoader, TensorDataset

def train_quick(model, a_tr, u_tr, a_te, u_te, n_epochs=15, lr=1e-3):
    u_mean = u_tr.mean().item()
    u_std  = u_tr.std().item()
    tr_ds  = TensorDataset(a_tr, u_tr)
    te_ds  = TensorDataset(a_te, u_te)
    tr_ld  = DataLoader(tr_ds, batch_size=20, shuffle=True)
    te_ld  = DataLoader(te_ds, batch_size=20)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    crit   = torch.nn.MSELoss()
    for _ in range(n_epochs):
        train_epoch(model, tr_ld, opt, crit, DEVICE, u_mean, u_std)
    return model, u_mean, u_std

def mode_error_map(model, a_te, u_te, u_mean, u_std, modes=MODES):
    model.eval()
    with torch.no_grad():
        pred = model(a_te)              
    pred_phys = pred * (u_std + 1e-8) + u_mean
    err = (pred_phys - u_te).unsqueeze(1)             
    err_fft = torch.fft.rfft2(err)
    _, _, H, W_r = err_fft.shape
    m2 = min(modes, W_r)
    m1 = min(modes, H)
    emap = np.zeros((m1, m2))
    for i in range(m1):
        for j in range(m2):
            emap[i, j] = torch.abs(err_fft[:, :, i, j]).mean().item()
    return emap

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Per-Mode Prediction Error $|\\hat{e}(k)|$: Standard vs. Condition-Aware',
             fontsize=13, fontweight='bold')

N_TR, N_TE = 150, 50

for col, (name, meta) in enumerate(DATASETS.items()):
    print(f'  Training {name} …')
    a, u = load_dataset(name, n=N_TR + N_TE)
    a_tr, u_tr = a[:N_TR], u[:N_TR]
    a_te, u_te = a[N_TR:], u[N_TR:]

                          
    std_model = FNO2d(modes1=MODES, modes2=MODES, width=32).to(DEVICE)
    std_model, um, us = train_quick(std_model, a_tr, u_tr, a_te, u_te)
    emap_std = mode_error_map(std_model, a_te, u_te, um, us)

                                 
    low_mask, high_mask = compute_adaptive_mask(a_tr, u_tr, MODES, MODES)
    ca_model = ConditionAwareFNO2d(low_mask, high_mask,
                                   modes1=MODES, modes2=MODES, width=32).to(DEVICE)
    ca_model, um, us = train_quick(ca_model, a_tr, u_tr, a_te, u_te)
    emap_ca = mode_error_map(ca_model, a_te, u_te, um, us)

    vmax = max(emap_std.max(), emap_ca.max())
    vmin = max(min(emap_std.min(), emap_ca.min()), 1e-10)

    for row, (emap, label, color) in enumerate([
        (emap_std, 'Standard FNO', C_STD),
        (emap_ca,  'Cond.-Aware',  C_CA),
    ]):
        ax = axes[row, col]
        im = ax.imshow(emap, cmap='hot', interpolation='nearest',
                       aspect='auto', vmin=vmin, vmax=vmax)
        if row == 0:
            ax.set_title(meta['label'], fontsize=11,
                         color=meta['color'], fontweight='bold')
        if col == 0:
            ax.set_ylabel(f'{label}\n$k_1$', fontsize=9, color=color)
        ax.set_xlabel('$k_2$', fontsize=8)
        ax.set_xticks(range(0, MODES, 3))
        if col > 0: ax.set_yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.04)
        fig.colorbar(im, cax=cax)

plt.tight_layout()
save(fig, '7_per_mode_error')

                                                                                 
                                                              
                                                                                 
print('Fig 8: η sensitivity …')

eta_vals = np.linspace(0.50, 0.995, 30)
fig, ax  = plt.subplots(figsize=(8, 5))
ax.set_title('Modes Retained vs. Energy Threshold $\\eta$', fontsize=12, fontweight='bold')

for name, meta in DATASETS.items():
    a, u  = load_dataset(name, n=100)
    retained = []
    for eta in eta_vals:
        lm, hm = compute_adaptive_mask(a, u, MODES, MODES, energy_fraction=float(eta))
        retained.append(int(lm.sum().item()))
    ax.plot(eta_vals, retained, color=meta['color'], linewidth=2.5, label=meta['label'])
    ax.scatter([0.95], [retained[np.argmin(np.abs(eta_vals - 0.95))]],
               color=meta['color'], s=80, zorder=5)

ax.axvline(x=0.95, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, label='η = 0.95 (used)')
ax.set_xlabel('Energy threshold $\\eta$', fontsize=11)
ax.set_ylabel('Modes retained (low-freq block)', fontsize=11)
ax.set_xlim(0.50, 1.0)
ax.set_ylim(0, MODES * MODES + 2)
ax.legend(fontsize=10)
plt.tight_layout()
save(fig, '8_eta_sensitivity')

print('\nAll figures saved to ./poster_figures/')
print('Files:')
for f in sorted(os.listdir('poster_figures')):
    print(f'  poster_figures/{f}')