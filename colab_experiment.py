# =====================================================================
#  Condition-Aware FNO — full experiment in ONE Colab cell
#  Paste this whole file into a single Colab cell and run.
#  Use a GPU runtime (Runtime > Change runtime type > GPU).
#
#  Outputs:
#    * prints LaTeX rows for the three result tables (paste into main.tex
#      between the RESULTS_*_BEGIN / RESULTS_*_END markers)
#    * saves  ca_results_main.csv, ca_results_ext.csv
#    * saves  benchmark_boxplot.png  and  convergence.png
#  If running in Colab the PNGs/CSVs are also offered as downloads.
# =====================================================================
import time, math, warnings, os, json, zipfile, platform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None
warnings.filterwarnings("ignore")

# ----------------------------- CONFIG --------------------------------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID        = 64
N_TRAIN     = 800
N_TEST      = 200
MODES       = 12
WIDTH       = 64
EPOCHS      = 50
BATCH       = 20
LR          = 1e-3
ETA         = 0.95          # energy-preservation fraction
MAIN_SEEDS  = 20            # seeds for the std-vs-CA comparison (lower to go faster)
EXT_SEEDS   = 6             # seeds for the extension subset
BILEVEL_ITERS, BILEVEL_LR = 80, 0.3
MAIN_DATASETS = ["poisson", "heat", "advection", "darcy", "wave", "darcy_multi", "heat_sensor"]
EXT_DATASETS  = ["heat", "wave", "darcy_multi", "heat_sensor"]
DIAG_DATASETS = ["heat_sensor", "wave", "darcy_multi"]   # per-mode kappa/energy/lambda diagnostics

# --------- SPLITTING ACROSS MULTIPLE NOTEBOOKS (run any subset) -------
# Each notebook is fully self-contained: it produces valid results for
# whatever it runs, written to ca_artifacts_<RUN_LABEL>/ and a matching zip,
# so parallel runs never collide.  Merge afterwards with combine_results.py
# (or just paste each notebook's printed LaTeX rows).  Examples:
#   Notebook A (main only):  RUN_LABEL="main"; RUN_MAIN=True;  RUN_EXT=False
#   Notebook B (ext only):   RUN_LABEL="ext";  RUN_MAIN=False; RUN_EXT=True
#   Split main by dataset:   RUN_LABEL="mainA"; MAIN_DATASETS=["poisson","heat","advection","darcy"]
#                            RUN_LABEL="mainB"; MAIN_DATASETS=["wave","darcy_multi","heat_sensor"]
RUN_LABEL = "full"     # MUST be distinct per notebook so downloads don't overwrite
RUN_MAIN  = True       # the std-vs-CA comparison table
RUN_EXT   = True       # the extensions table (now includes the SNR-gated model)
RUN_DIAG  = True       # per-mode diagnostics (only for DIAG_DATASETS this notebook touches)

# --------- SNR-GATED FNO (the optimal-shrinkage method, Sec. 7) -------
# Replaces hard truncation with a soft per-mode Wiener gate g*(k)=SNR/(1+SNR),
# estimated as the per-mode held-out R^2 of the diagonal ridge fit, applied to
# the FULL rectangular mode box (no truncation -> never starves capacity).
# g*->1 on clean data (ties std FNO); g*<1 shrinks unreliable modes (denoises).
GATE_LEARNABLE = False   # False: fixed plug-in Wiener gate; True: learnable, warm-started from g*
GATE_EPS       = 1e-3    # floor on the gate so a mode is never fully removed
GATE_L1        = 0.0     # optional sparsity penalty on learnable gates (0 = off)

# --------- DATA-BUDGET / NOISE SWEEP (where shrinkage provably wins) ---
# Trains std vs SNR-gated FNO under shrinking N and growing observation noise,
# the regime in which Theorem 7.3 predicts a strict win (V is non-negligible).
RUN_SWEEP     = True
SWEEP_DATASETS = ["heat", "advection"]      # clean diagonal operators: cleanest test
SWEEP_N        = [100, 400, 800]            # training-set sizes
SWEEP_NOISE    = [0.0, 0.1, 0.2]            # observation-noise std (fraction of field std)
SWEEP_SEEDS    = 3
SWEEP_MODELS   = ["std", "snr_fixed", "snr_learn"]

SAVE_CHECKPOINTS = True      # save trained model weights (one seed) — see filters below
CKPT_SEED        = 42        # which seed to checkpoint
CKPT_MODELS      = {"std", "ca", "snr_fixed", "snr_learn"}   # which model types to checkpoint (each ~38 MB at full config)
CKPT_DATASETS    = ["heat", "darcy", "wave", "darcy_multi", "heat_sensor"]  # which datasets to checkpoint
OUTDIR           = f"ca_artifacts_{RUN_LABEL}"
print("Device:", DEVICE, "| run label:", RUN_LABEL)

# ----- output tree --------------------------------------------------
for sub in ["", "curves", "checkpoints", "diagnostics", "figures", "logs"]:
    os.makedirs(os.path.join(OUTDIR, sub), exist_ok=True)
ALL_CURVES = {}   # (dataset, model, seed) -> {"test_relL2": [...], "train_mse": [...]}
RUN_META = {
    "device": str(DEVICE), "torch": torch.__version__, "platform": platform.platform(),
    "config": {k: v for k, v in dict(GRID=GRID, N_TRAIN=N_TRAIN, N_TEST=N_TEST, MODES=MODES,
               WIDTH=WIDTH, EPOCHS=EPOCHS, BATCH=BATCH, LR=LR, ETA=ETA,
               MAIN_SEEDS=MAIN_SEEDS, EXT_SEEDS=EXT_SEEDS,
               BILEVEL_ITERS=BILEVEL_ITERS, BILEVEL_LR=BILEVEL_LR).items()},
    "started": time.strftime("%Y-%m-%d %H:%M:%S"),
}

TWO_PI = 2.0 * np.pi

# =====================================================================
#  DATASETS  (genuine PDE solution operators; input determines output)
# =====================================================================
def _wavenumbers(n):
    k = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return kx, ky, kx**2 + ky**2

def _grf(n_samples, n, alpha, rng):
    _, _, k2 = _wavenumbers(n); kmag = np.sqrt(k2)
    amp = np.zeros_like(kmag); nz = kmag > 0
    amp[nz] = kmag[nz] ** (-alpha / 2.0)
    noise = rng.standard_normal((n_samples, n, n)) + 1j*rng.standard_normal((n_samples, n, n))
    f = np.fft.ifft2(amp[None]*noise, axes=(-2, -1)).real
    f -= f.mean((-2, -1), keepdims=True)
    return f / (f.std((-2, -1), keepdims=True) + 1e-8)

def _pack(a, u):
    if a.ndim == 3: a = a[..., None]
    return TensorDataset(torch.from_numpy(np.ascontiguousarray(a)).float(),
                         torch.from_numpy(np.ascontiguousarray(u)).float())

def _solve_darcy(a, f, n_iter=400, tol=1e-6):
    """ -div(a grad u)=f on periodic torus via matrix-free PCG (FFT preconditioner)."""
    s, n, _ = a.shape; dx2 = (1.0/n)**2
    harm = lambda x, y: 2.0*x*y/(x+y+1e-30)
    axp, axm = harm(a, np.roll(a,-1,-2)), harm(a, np.roll(a,1,-2))
    ayp, aym = harm(a, np.roll(a,-1,-1)), harm(a, np.roll(a,1,-1))
    def A(u):
        return (axp*(u-np.roll(u,-1,-2))+axm*(u-np.roll(u,1,-2))
               +ayp*(u-np.roll(u,-1,-1))+aym*(u-np.roll(u,1,-1)))/dx2
    a_bar = a.mean((-2,-1), keepdims=True)
    kk = np.arange(n); lam = (2-2*np.cos(TWO_PI*kk/n))/dx2
    L = lam[:,None]+lam[None,:]; inv = np.zeros_like(L); inv[L>0]=1.0/L[L>0]; inv=inv[None]
    zm = lambda v: v - v.mean((-2,-1), keepdims=True)
    P  = lambda r: np.fft.ifft2(np.fft.fft2(r,axes=(-2,-1))*inv,axes=(-2,-1)).real/a_bar
    dot= lambda p,q: np.sum(p*q,(-2,-1),keepdims=True)
    f = zm(f); fn = np.sqrt(dot(f,f))+1e-30
    x=np.zeros_like(f); r=f.copy(); z=zm(P(r)); p=z.copy(); rz=dot(r,z)
    for _ in range(n_iter):
        Ap=zm(A(p)); al=rz/(dot(p,Ap)+1e-30); x+=al*p; r-=al*Ap
        if np.max(np.sqrt(dot(r,r))/fn) < tol: break
        z=zm(P(r)); rzn=dot(r,z); p=z+(rzn/(rz+1e-30))*p; rz=rzn
    return zm(x)

def make_poisson(ns, n, rng):
    _,_,k2=_wavenumbers(n); inv=np.zeros_like(k2); nz=k2>0; inv[nz]=1/(TWO_PI**2*k2[nz])
    f=_grf(ns,n,2.0,rng); u=np.fft.ifft2(np.fft.fft2(f,axes=(-2,-1))*inv[None],axes=(-2,-1)).real
    return _pack(f,u)

def make_heat(ns, n, rng, nu=0.01, T=0.05):
    _,_,k2=_wavenumbers(n); d=np.exp(-nu*TWO_PI**2*k2*T)
    u0=_grf(ns,n,1.0,rng); uT=np.fft.ifft2(np.fft.fft2(u0,axes=(-2,-1))*d[None],axes=(-2,-1)).real
    return _pack(u0,uT)

def make_advection(ns, n, rng, bx=1.0, by=0.6, nu=0.01, T=0.1):
    kx,ky,k2=_wavenumbers(n); sym=np.exp(-(1j*TWO_PI*(bx*kx+by*ky)+nu*TWO_PI**2*k2)*T)
    u0=_grf(ns,n,1.5,rng); uT=np.fft.ifft2(np.fft.fft2(u0,axes=(-2,-1))*sym[None],axes=(-2,-1)).real
    return _pack(u0,uT)

def make_wave(ns, n, rng, c=1.0, T=0.2):
    _,_,k2=_wavenumbers(n); km=np.sqrt(k2); w=c*TWO_PI*km
    cos=np.cos(w*T); sinc=np.where(w>0, np.sin(w*T)/np.where(w>0,w,1.0), T)
    u0=_grf(ns,n,1.5,rng); v0=_grf(ns,n,1.5,rng)
    uT=np.fft.ifft2(cos[None]*np.fft.fft2(u0,axes=(-2,-1))+sinc[None]*np.fft.fft2(v0,axes=(-2,-1)),axes=(-2,-1)).real
    return _pack(np.stack([u0,v0],-1), uT)

def make_darcy(ns, n, rng, contrast=4.0):
    a=np.exp(0.5*np.log(contrast)*_grf(ns,n,2.0,rng))
    xs=np.sin(TWO_PI*np.linspace(0,1,n,endpoint=False))
    cs=np.cos(TWO_PI*np.linspace(0,1,n,endpoint=False))
    f=np.broadcast_to(xs[None,:,None],(ns,n,n)).copy()+cs[None,None,:]
    return _pack(a, _solve_darcy(a,f))

def make_darcy_multi(ns, n, rng, contrast=4.0):
    a=np.exp(0.5*np.log(contrast)*_grf(ns,n,2.0,rng)); f=_grf(ns,n,1.5,rng)
    return _pack(np.stack([a,f],-1), _solve_darcy(a,f))

def make_heat_sensor(ns, n, rng, noise=0.15, nu=0.01, T=0.05):
    _,_,k2=_wavenumbers(n); d=np.exp(-nu*TWO_PI**2*k2*T)
    u0=_grf(ns,n,1.0,rng)
    uT=np.fft.ifft2(np.fft.fft2(u0,axes=(-2,-1))*d[None],axes=(-2,-1)).real
    a=np.stack([u0+noise*rng.standard_normal((ns,n,n)), u0+noise*rng.standard_normal((ns,n,n))],-1)
    return _pack(a, uT)

BUILDERS = {"poisson":make_poisson,"heat":make_heat,"advection":make_advection,
            "darcy":make_darcy,"wave":make_wave,"darcy_multi":make_darcy_multi,
            "heat_sensor":make_heat_sensor}
IN_CH    = {"poisson":1,"heat":1,"advection":1,"darcy":1,"wave":2,"darcy_multi":2,"heat_sensor":2}

def get_dataset(name, ns, seed):
    return BUILDERS[name](ns, GRID, np.random.default_rng(seed))

# =====================================================================
#  SCORING / MASK / RIDGE  (precompute on CPU — robust across backends)
# =====================================================================
def _to_cf(x):
    if x.dim()==3: return x.unsqueeze(1)
    return x.permute(0,3,1,2).contiguous()

def _energy_cond(X, Y):
    e = torch.sum(torch.abs(Y)**2).real.item()
    try:
        s = torch.linalg.svdvals(X.detach().cpu())
        if s.numel() and s[-1].item()>1e-9: c=(s[0]/s[-1]).item()
        elif s.numel() and s[0].item()>1e-12: c=1e9
        else: c=1.0
    except Exception: c=1e9
    return e, c

def _select(scores, energies, eta):
    fs, fe = scores.reshape(-1), energies.reshape(-1); tot=fe.sum()
    if tot<=0: return torch.ones_like(scores, dtype=torch.bool)
    order=torch.argsort(fs, descending=True); cum=torch.cumsum(fe[order],0)
    k=int((cum < eta*tot).sum().item())+1; k=max(1,min(k,fs.numel()))
    m=torch.zeros_like(fs, dtype=torch.bool); m[order[:k]]=True
    return m.view_as(scores)

def compute_adaptive_mask(a, u, modes1, modes2, eta=ETA):
    af=torch.fft.rfft2(_to_cf(a)); uf=torch.fft.rfft2(_to_cf(u))
    _,_,Hf,Wf=af.shape; m2=min(modes2,Wf)
    def met(i0,i1):
        m1=max(i1-i0,0); sc=torch.zeros(m1,modes2); en=torch.zeros(m1,modes2)
        for i in range(i0,i1):
            for j in range(m2):
                e,c=_energy_cond(af[:,:,i,j], uf[:,:,i,j]); en[i-i0,j]=e; sc[i-i0,j]=e/(c+1e-8)
        return sc,en
    ls,le=met(0,min(modes1,Hf)); hs,he=met(max(Hf-modes1,0),Hf)
    return _select(ls,le,eta), _select(hs,he,eta)

def compute_full_spectrum_mask(a, u, Hf, Wf, eta=ETA):
    af=torch.fft.rfft2(_to_cf(a)); uf=torch.fft.rfft2(_to_cf(u)); half=Hf//2
    def blk(i0,i1):
        m1=i1-i0; sc=torch.zeros(m1,Wf); en=torch.zeros(m1,Wf)
        for i in range(i0,i1):
            for j in range(Wf):
                e,c=_energy_cond(af[:,:,i,j], uf[:,:,i,j]); en[i-i0,j]=e; sc[i-i0,j]=e/(c+1e-8)
        return sc,en
    ls,le=blk(0,half); hs,he=blk(Hf-half,Hf)
    return _select(ls,le,eta), _select(hs,he,eta)

def compute_learnable_lambda(atr,utr,aval,uval,modes1,modes2,n_iters=BILEVEL_ITERS,lr=BILEVEL_LR,eta=ETA):
    aft,uft=torch.fft.rfft2(_to_cf(atr)),torch.fft.rfft2(_to_cf(utr))
    avf,uvf=torch.fft.rfft2(_to_cf(aval)),torch.fft.rfft2(_to_cf(uval))
    _,_,Hf,Wf=aft.shape; m2=min(modes2,Wf)
    def region(i0,i1):
        m1=max(i1-i0,0); lam=torch.zeros(m1,modes2); sc=torch.zeros(m1,modes2); en=torch.zeros(m1,modes2)
        for i in range(i0,i1):
            for j in range(m2):
                Xt,Yt=aft[:,:,i,j],uft[:,:,i,j]; Xv,Yv=avf[:,:,i,j],uvf[:,:,i,j]
                d=Xt.shape[1]; XHX=Xt.conj().T@Xt; B=Xt.conj().T@Yt; I=torch.eye(d,dtype=XHX.dtype)
                a_=0.0
                for _ in range(n_iters):
                    l=math.exp(a_);
                    try: Ai=torch.linalg.inv(XHX+l*I)
                    except Exception: break
                    Cs=Ai@B; R=Xv@Cs-Yv; dC=-l*(Ai@Cs)
                    g=2.0*torch.sum((Xv@dC).conj()*R).real.item()
                    a_=max(min(a_-lr*g,12.0),-12.0)
                lam[i-i0,j]=math.exp(a_)
                e,c=_energy_cond(Xt,Yt); en[i-i0,j]=e; sc[i-i0,j]=e/(c+1e-8)
        return lam,sc,en
    ll,ls,le=region(0,min(modes1,Hf)); hl,hs,he=region(max(Hf-modes1,0),Hf)
    return ll,hl,_select(ls,le,eta),_select(hs,he,eta)

# =====================================================================
#  MODELS
# =====================================================================
class SpectralConv2d(nn.Module):
    def __init__(s,ci,co,m1,m2):
        super().__init__(); s.ci,s.co,s.m1,s.m2=ci,co,m1,m2; sc=1/(ci*co)
        s.w1=nn.Parameter(sc*torch.rand(ci,co,m1,m2,dtype=torch.cfloat))
        s.w2=nn.Parameter(sc*torch.rand(ci,co,m1,m2,dtype=torch.cfloat))
    def mul(s,x,w): return torch.einsum("bixy,ioxy->boxy",x,w)
    def forward(s,x):
        B,C,H,W=x.shape; xf=torch.fft.rfft2(x); _,_,Hf,Wf=xf.shape
        o=torch.zeros(B,s.co,Hf,Wf,dtype=torch.cfloat,device=x.device)
        m1,m2=min(s.m1,Hf),min(s.m2,Wf)
        o[:,:,:m1,:m2]=s.mul(xf[:,:,:m1,:m2],s.w1[:,:,:m1,:m2])
        o[:,:,-m1:,:m2]=s.mul(xf[:,:,-m1:,:m2],s.w2[:,:,:m1,:m2])
        return torch.fft.irfft2(o,s=(H,W))

class CASpectralConv2d(nn.Module):
    def __init__(s,ci,co,lm,hm,m1,m2,ll=None,hl=None):
        super().__init__(); s.ci,s.co,s.m1,s.m2=ci,co,m1,m2
        s.register_buffer("lm",lm.bool()); s.register_buffer("hm",hm.bool())
        z=torch.zeros(m1,m2)
        s.register_buffer("ll", z.clone() if ll is None else ll.float())
        s.register_buffer("hl", z.clone() if hl is None else hl.float())
        sc=1/(ci*co)
        s.w1=nn.Parameter(sc*torch.rand(ci,co,m1,m2,dtype=torch.cfloat))
        s.w2=nn.Parameter(sc*torch.rand(ci,co,m1,m2,dtype=torch.cfloat))
    def mul(s,x,w): return torch.einsum("bixy,ioxy->boxy",x,w)
    def forward(s,x):
        B,C,H,W=x.shape; xf=torch.fft.rfft2(x); _,_,Hf,Wf=xf.shape
        o=torch.zeros(B,s.co,Hf,Wf,dtype=torch.cfloat,device=x.device)
        m1,m2=min(s.m1,Hf),min(s.m2,Wf)
        lm=s.lm[:m1,:m2].to(torch.cfloat); hm=s.hm[:m1,:m2].to(torch.cfloat)
        w1=s.w1[:,:,:m1,:m2]*lm[None,None]; w2=s.w2[:,:,:m1,:m2]*hm[None,None]
        o[:,:,:m1,:m2]=s.mul(xf[:,:,:m1,:m2],w1); o[:,:,-m1:,:m2]=s.mul(xf[:,:,-m1:,:m2],w2)
        return torch.fft.irfft2(o,s=(H,W))
    def reg(s):
        w1=(s.w1.abs()**2).sum((0,1)); w2=(s.w2.abs()**2).sum((0,1))
        return (s.ll*s.lm.float()*w1).sum()+(s.hl*s.hm.float()*w2).sum()

class _Backbone(nn.Module):
    """shared head/tail; subclasses set s.convs (list of 4 spectral layers)."""
    def _build(s, in_ch, width):
        s.fc0=nn.Linear(in_ch,width)
        s.w0=nn.Conv2d(width,width,1); s.w1=nn.Conv2d(width,width,1)
        s.w2=nn.Conv2d(width,width,1); s.w3=nn.Conv2d(width,width,1)
        s.fc1=nn.Linear(width,128); s.fc2=nn.Linear(128,1)
    def forward(s,x):
        if x.dim()==3: x=x.unsqueeze(-1)
        x=s.fc0(x).permute(0,3,1,2)
        x=F.gelu(s.conv0(x)+s.w0(x)); x=F.gelu(s.conv1(x)+s.w1(x))
        x=F.gelu(s.conv2(x)+s.w2(x)); x=s.conv3(x)+s.w3(x)
        x=x.permute(0,2,3,1); x=F.gelu(s.fc1(x)); return s.fc2(x).squeeze(-1)
    def spectral_regularization(s):
        return sum(c.reg() for c in [s.conv0,s.conv1,s.conv2,s.conv3] if hasattr(c,"reg"))

class FNO2d(_Backbone):
    def __init__(s,in_ch=1,m1=MODES,m2=MODES,width=WIDTH):
        super().__init__(); s.m1,s.m2,s.width=m1,m2,width; s._build(in_ch,width)
        s.conv0,s.conv1,s.conv2,s.conv3=[SpectralConv2d(width,width,m1,m2) for _ in range(4)]

class CAFNO2d(_Backbone):
    def __init__(s,lm,hm,in_ch=1,m1=MODES,m2=MODES,width=WIDTH,ll=None,hl=None):
        super().__init__(); s.m1,s.m2,s.width=m1,m2,width; s._build(in_ch,width)
        mk=lambda: CASpectralConv2d(width,width,lm,hm,m1,m2,ll,hl)
        s.conv0,s.conv1,s.conv2,s.conv3=mk(),mk(),mk(),mk()

class DynamicCAFNO2d(CAFNO2d):
    def update_from_residuals(s,a,u,um,ustd,eta=ETA):
        s.eval()
        with torch.no_grad(): pred=s(a)
        r=((u-um)/(ustd+1e-8))-pred
        af=torch.fft.rfft2(_to_cf(a)); rf=torch.fft.rfft2(_to_cf(r))
        _,_,Hf,Wf=af.shape; m2=min(s.m2,Wf)
        def met(i0,i1):
            m1=max(i1-i0,0); sc=torch.zeros(m1,s.m2); en=torch.zeros(m1,s.m2)
            for i in range(i0,i1):
                for j in range(m2):
                    e,c=_energy_cond(af[:,:,i,j],rf[:,:,i,j]); en[i-i0,j]=e; sc[i-i0,j]=e/(c+1e-8)
            return sc,en
        ls,le=met(0,min(s.m1,Hf)); hs,he=met(max(Hf-s.m1,0),Hf)
        nl,nh=_select(ls,le,eta).to(a.device),_select(hs,he,eta).to(a.device)
        for c in [s.conv0,s.conv1,s.conv2,s.conv3]: c.lm.copy_(nl); c.hm.copy_(nh)

class AnisoSpectralConv2d(nn.Module):
    def __init__(s,ci,co,lm,hm):
        super().__init__(); s.ci,s.co=ci,co; half,Wf=lm.shape; s.half,s.Wf=half,Wf
        s.register_buffer("lm",lm.bool()); s.register_buffer("hm",hm.bool()); sc=1/(ci*co)
        s.w1=nn.Parameter(sc*torch.rand(ci,co,half,Wf,dtype=torch.cfloat))
        s.w2=nn.Parameter(sc*torch.rand(ci,co,half,Wf,dtype=torch.cfloat))
    def forward(s,x):
        B,C,H,W=x.shape; xf=torch.fft.rfft2(x); _,_,Hf,Wfa=xf.shape
        half=min(s.half,Hf//2); wf=min(s.Wf,Wfa)
        o=torch.zeros(B,s.co,Hf,Wfa,dtype=torch.cfloat,device=x.device)
        lm=s.lm[:half,:wf].to(torch.cfloat); hm=s.hm[:half,:wf].to(torch.cfloat)
        w1=s.w1[:,:,:half,:wf]*lm[None,None]; w2=s.w2[:,:,:half,:wf]*hm[None,None]
        o[:,:,:half,:wf]=torch.einsum("bixy,ioxy->boxy",xf[:,:,:half,:wf],w1)
        o[:,:,-half:,:wf]=torch.einsum("bixy,ioxy->boxy",xf[:,:,-half:,:wf],w2)
        return torch.fft.irfft2(o,s=(H,W))

class AnisoCAFNO2d(_Backbone):
    def __init__(s,lm,hm,in_ch=1,width=WIDTH):
        super().__init__(); s.width=width; s._build(in_ch,width)
        mk=lambda: AnisoSpectralConv2d(width,width,lm,hm)
        s.conv0,s.conv1,s.conv2,s.conv3=mk(),mk(),mk(),mk()

class PerLayerCAFNO2d(CAFNO2d):
    @torch.no_grad()
    def bootstrap(s, loader, eta=ETA, n_batches=20):
        s.eval(); acts={i:[] for i in range(5)}; convs=[s.conv0,s.conv1,s.conv2,s.conv3]
        def hook(idx):
            def h(m,inp,out):
                acts[idx].append(inp[0].detach().cpu())
                if idx==3: acts[4].append(out.detach().cpu())
            return h
        hs=[c.register_forward_hook(hook(i)) for i,c in enumerate(convs)]
        for bi,(a,_) in enumerate(loader):
            if bi>=n_batches: break
            s(a.to(next(s.parameters()).device))
        for h in hs: h.remove()
        for li in range(4):
            fi=torch.cat(acts[li],0); fo=torch.cat(acts[li+1],0)
            af=torch.fft.rfft2(fi); uf=torch.fft.rfft2(fo); _,_,Hf,Wf=af.shape; m2=min(s.m2,Wf)
            def met(i0,i1):
                m1=max(i1-i0,0); sc=torch.zeros(m1,s.m2); en=torch.zeros(m1,s.m2)
                for i in range(i0,i1):
                    for j in range(m2):
                        e,c=_energy_cond(af[:,:,i,j],uf[:,:,i,j]); en[i-i0,j]=e; sc[i-i0,j]=e/(c+1e-8)
                return sc,en
            ls,le=met(0,min(s.m1,Hf)); hs2,he=met(max(Hf-s.m1,0),Hf)
            dev=next(s.parameters()).device
            convs[li].lm.copy_(_select(ls,le,eta).to(dev)); convs[li].hm.copy_(_select(hs2,he,eta).to(dev))

# =====================================================================
#  SNR-GATED FNO  (optimal per-mode shrinkage; Sec. 7 of the writeup)
# =====================================================================
def compute_snr_prior(a_fit, u_fit, a_val, u_val, modes1, modes2, ridge=1e-3):
    """Per-mode optimal-shrinkage gate g*(k) = held-out R^2 of the diagonal
    ridge fit = SNR/(1+SNR).  Returns (g_low, g_high), each (modes1, modes2),
    clipped to [GATE_EPS, 1].  Estimated on CPU (complex linalg)."""
    af = torch.fft.rfft2(_to_cf(a_fit)); uf = torch.fft.rfft2(_to_cf(u_fit))
    avf = torch.fft.rfft2(_to_cf(a_val)); uvf = torch.fft.rfft2(_to_cf(u_val))
    _, Cin, Hf, Wf = af.shape; m2 = min(modes2, Wf)

    def region(i0, i1):
        m1 = max(i1 - i0, 0); g = torch.full((m1, modes2), GATE_EPS)
        for i in range(i0, i1):
            for j in range(m2):
                Xt, Yt = af[:, :, i, j], uf[:, :, i, j]
                Xv, Yv = avf[:, :, i, j], uvf[:, :, i, j]
                d = Xt.shape[1]
                A = Xt.conj().T @ Xt + ridge * torch.eye(d, dtype=Xt.dtype)
                C = torch.linalg.solve(A, Xt.conj().T @ Yt)
                ss_res = ((Xv @ C - Yv).abs() ** 2).sum()
                ss_tot = (Yv.abs() ** 2).sum() + 1e-12
                r2 = 1.0 - (ss_res / ss_tot).real.item()
                g[i - i0, j] = min(max(r2, GATE_EPS), 1.0)
        return g
    return region(0, min(modes1, Hf)), region(max(Hf - modes1, 0), Hf)


class SNRGatedSpectralConv2d(nn.Module):
    """Full-box spectral conv with a per-mode shrinkage gate (no truncation).

    out(k) = g(k) * (M(k) . in(k)).  g is either a fixed buffer (plug-in Wiener
    gate g*) or a learnable logit warm-started from g*.  g==1 recovers the
    standard spectral conv exactly.
    """
    def __init__(s, ci, co, m1, m2, g_low, g_high, learnable=False):
        super().__init__(); s.ci, s.co, s.m1, s.m2 = ci, co, m1, m2
        sc = 1.0 / (ci * co)
        s.w1 = nn.Parameter(sc * torch.rand(ci, co, m1, m2, dtype=torch.cfloat))
        s.w2 = nn.Parameter(sc * torch.rand(ci, co, m1, m2, dtype=torch.cfloat))
        s.learnable = learnable
        if learnable:
            logit = lambda g: torch.log(g.clamp(GATE_EPS, 1 - 1e-4) / (1 - g.clamp(GATE_EPS, 1 - 1e-4)))
            s.glow = nn.Parameter(logit(g_low)); s.ghigh = nn.Parameter(logit(g_high))
        else:
            s.register_buffer("glow", g_low.clone()); s.register_buffer("ghigh", g_high.clone())

    def _g(s):
        return (torch.sigmoid(s.glow), torch.sigmoid(s.ghigh)) if s.learnable else (s.glow, s.ghigh)

    def mul(s, x, w): return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(s, x):
        B, C, H, W = x.shape; xf = torch.fft.rfft2(x); _, _, Hf, Wf = xf.shape
        o = torch.zeros(B, s.co, Hf, Wf, dtype=torch.cfloat, device=x.device)
        m1, m2 = min(s.m1, Hf), min(s.m2, Wf)
        gl, gh = s._g()
        gl = gl[:m1, :m2].to(x.device); gh = gh[:m1, :m2].to(x.device)
        o[:, :, :m1, :m2] = s.mul(xf[:, :, :m1, :m2], s.w1[:, :, :m1, :m2]) * gl[None, None]
        o[:, :, -m1:, :m2] = s.mul(xf[:, :, -m1:, :m2], s.w2[:, :, :m1, :m2]) * gh[None, None]
        return torch.fft.irfft2(o, s=(H, W))

    def gate_l1(s):
        gl, gh = s._g(); return gl.sum() + gh.sum()

    @torch.no_grad()
    def gate_mean(s):
        gl, gh = s._g(); return float((gl.mean() + gh.mean()) / 2)


class SNRGatedFNO2d(_Backbone):
    """FNO with full mode box and per-mode shrinkage gates at every layer."""
    def __init__(s, g_low, g_high, in_ch=1, m1=MODES, m2=MODES, width=WIDTH, learnable=GATE_LEARNABLE):
        super().__init__(); s.m1, s.m2, s.width = m1, m2, width; s._build(in_ch, width)
        mk = lambda: SNRGatedSpectralConv2d(width, width, m1, m2, g_low, g_high, learnable)
        s.conv0, s.conv1, s.conv2, s.conv3 = mk(), mk(), mk(), mk()
        s.learnable = learnable

    def spectral_regularization(s):
        if not (s.learnable and GATE_L1 > 0):
            return torch.zeros((), device=next(s.parameters()).device)
        return GATE_L1 * sum(c.gate_l1() for c in [s.conv0, s.conv1, s.conv2, s.conv3])

    @torch.no_grad()
    def mean_gate(s):
        """Mean gate value across layers and modes (final, post-training)."""
        return float(np.mean([c.gate_mean() for c in [s.conv0, s.conv1, s.conv2, s.conv3]]))


# =====================================================================
#  TRAIN / EVAL  (objective: MSE on standardized targets; metric: rel L2)
# =====================================================================
def train_epoch(model, loader, opt, um, ustd):
    model.train(); crit=nn.MSELoss(); reg=hasattr(model,"spectral_regularization")
    tot=0.0; n=0
    for a,u in loader:
        a,u=a.to(DEVICE),u.to(DEVICE); un=(u-um)/(ustd+1e-8)
        opt.zero_grad(); loss=crit(model(a),un)
        if reg: loss=loss+model.spectral_regularization()
        loss.backward(); opt.step()
        tot+=loss.item()*a.size(0); n+=a.size(0)
    return tot/max(n,1)

@torch.no_grad()
def eval_relL2(model, loader, um, ustd):
    model.eval(); tot=0.0; n=0
    for a,u in loader:
        a,u=a.to(DEVICE),u.to(DEVICE); un=(u-um)/(ustd+1e-8); pred=model(a)
        num=torch.linalg.vector_norm((pred-un).reshape(a.size(0),-1),dim=1)
        den=torch.linalg.vector_norm(un.reshape(a.size(0),-1),dim=1)+1e-8
        tot+=(num/den).sum().item(); n+=a.size(0)
    return tot/n

def set_seed(s): torch.manual_seed(s); np.random.seed(s)

def train_model(model, trl, tel, um, ustd, dyn=None):
    """Train; always record per-epoch test rel-L2 and train MSE. Returns
    (final_relL2, elapsed_s, model, test_curve, train_curve)."""
    model=model.to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=LR)
    test_c, train_c = [], []
    t0=time.time()
    sched=set(int(EPOCHS*0.6**(4-1-i)) for i in range(4)) if dyn is not None else set()
    for ep in range(EPOCHS):
        if ep in sched and dyn is not None:
            a_u,u_u=dyn; model.update_from_residuals(a_u,u_u,um,ustd)
        tr=train_epoch(model, trl, opt, um, ustd)
        train_c.append(tr); test_c.append(eval_relL2(model, tel, um, ustd))
    return test_c[-1], time.time()-t0, model, test_c, train_c

def _save_ckpt(model, ds, name, seed):
    if SAVE_CHECKPOINTS and seed == CKPT_SEED and name in CKPT_MODELS and ds in CKPT_DATASETS:
        torch.save({"state_dict": model.state_dict(), "dataset": ds, "model": name,
                    "config": RUN_META["config"]},
                   os.path.join(OUTDIR, "checkpoints", f"{ds}_{name}_seed{seed}.pt"))

def _record(ds, name, seed, test_c, train_c):
    ALL_CURVES[f"{ds}|{name}|{seed}"] = {"test_relL2": test_c, "train_mse": train_c}

# =====================================================================
#  BENCHMARK
# =====================================================================
def run(datasets, models, seeds):
    rows=[]
    for ds in datasets:
        inc=IN_CH[ds]
        for seed in range(42, 42+seeds):
            set_seed(seed)
            tr=get_dataset(ds,N_TRAIN,seed); te=get_dataset(ds,N_TEST,seed+1000)
            trl=DataLoader(tr,batch_size=BATCH,shuffle=True)
            tel=DataLoader(te,batch_size=BATCH,shuffle=False)
            A=torch.stack([tr[i][0] for i in range(len(tr))])
            U=torch.stack([tr[i][1] for i in range(len(tr))])
            um,ustd=U.mean().item(),U.std().item(); un=(U-um)/(ustd+1e-8)
            Am,Um=A[:200].cpu(),un[:200].cpu()
            row={"Dataset":ds,"Seed":seed}
            lm=hm=None
            if any(m in models for m in ["ca","ll","per_layer","dynamic"]):
                lm,hm=compute_adaptive_mask(Am,Um,MODES,MODES)
            if any(m in models for m in ["snr_fixed","snr_learn"]):
                h=len(Am)//2
                g_low,g_high=compute_snr_prior(Am[:h],Um[:h],Am[h:],Um[h:],MODES,MODES)
                row["gate_prior_mean"]=float((g_low.mean()+g_high.mean())/2)
            if "std" in models:
                set_seed(seed); l,t,m,tc,trc=train_model(FNO2d(inc),trl,tel,um,ustd)
                row["std_loss"],row["std_time"]=l,t; _record(ds,"std",seed,tc,trc); _save_ckpt(m,ds,"std",seed)
            if "snr_fixed" in models:
                set_seed(seed); m=SNRGatedFNO2d(g_low,g_high,inc,learnable=False)
                l,t,m,tc,trc=train_model(m,trl,tel,um,ustd)
                row["snr_fixed_loss"],row["snr_fixed_time"]=l,t; _record(ds,"snr_fixed",seed,tc,trc); _save_ckpt(m,ds,"snr_fixed",seed)
                row["snr_fixed_gate_final"]=m.mean_gate()
            if "snr_learn" in models:
                set_seed(seed); m=SNRGatedFNO2d(g_low,g_high,inc,learnable=True)
                l,t,m,tc,trc=train_model(m,trl,tel,um,ustd)
                row["snr_learn_loss"],row["snr_learn_time"]=l,t; _record(ds,"snr_learn",seed,tc,trc); _save_ckpt(m,ds,"snr_learn",seed)
                row["snr_learn_gate_final"]=m.mean_gate()
            if "ca" in models:
                set_seed(seed); l,t,m,tc,trc=train_model(CAFNO2d(lm,hm,inc),trl,tel,um,ustd)
                row["ca_loss"],row["ca_time"]=l,t; _record(ds,"ca",seed,tc,trc); _save_ckpt(m,ds,"ca",seed)
                row["ca_modes_low"]=int(lm.sum().item()); row["ca_modes_high"]=int(hm.sum().item())
            if "ll" in models:
                tll=time.time()
                ll_,hl_,lmll,hmll=compute_learnable_lambda(Am[:100],Um[:100],Am[100:],Um[100:],MODES,MODES)
                def norm(x): mm=x.mean(); return (x/(mm+1e-12) if mm>0 else torch.ones_like(x))*1e-3
                pre=time.time()-tll
                set_seed(seed); l,t,m,tc,trc=train_model(CAFNO2d(lmll,hmll,inc,ll=norm(ll_),hl=norm(hl_)),trl,tel,um,ustd)
                row["ll_loss"],row["ll_time"]=l,t+pre; row["ll_precomp_time"]=pre
                _record(ds,"ll",seed,tc,trc); _save_ckpt(m,ds,"ll",seed)
            if "per_layer" in models:
                set_seed(seed); m=PerLayerCAFNO2d(lm,hm,inc).to(DEVICE)
                opt=torch.optim.Adam(m.parameters(),lr=LR)
                for _ in range(max(1,EPOCHS//10)): train_epoch(m,trl,opt,um,ustd)
                m.bootstrap(trl)
                l,t,m,tc,trc=train_model(m,trl,tel,um,ustd)
                row["per_layer_loss"],row["per_layer_time"]=l,t; _record(ds,"per_layer",seed,tc,trc); _save_ckpt(m,ds,"per_layer",seed)
            if "dynamic" in models:
                set_seed(seed); m=DynamicCAFNO2d(lm,hm,inc)
                au=torch.cat([A[i*BATCH:(i+1)*BATCH] for i in range(min(20,len(tr)//BATCH))]).to(DEVICE)
                uu=torch.cat([U[i*BATCH:(i+1)*BATCH] for i in range(min(20,len(tr)//BATCH))]).to(DEVICE)
                l,t,m,tc,trc=train_model(m,trl,tel,um,ustd,dyn=(au,uu))
                row["dynamic_loss"],row["dynamic_time"]=l,t; _record(ds,"dynamic",seed,tc,trc); _save_ckpt(m,ds,"dynamic",seed)
            if "anisotropic" in models:
                flm,fhm=compute_full_spectrum_mask(Am,Um,GRID,GRID//2+1)
                set_seed(seed); l,t,m,tc,trc=train_model(AnisoCAFNO2d(flm,fhm,inc),trl,tel,um,ustd)
                row["aniso_loss"],row["aniso_time"]=l,t; _record(ds,"aniso",seed,tc,trc); _save_ckpt(m,ds,"aniso",seed)
            rows.append(row)
            msg=" ".join(f"{k.split('_')[0]}={row[k]:.4f}" for k in row if k.endswith("_loss"))
            print(f"[{ds} seed={seed}] {msg}")
    return pd.DataFrame(rows)

EMPTY = pd.DataFrame(columns=["Dataset", "Seed"])
if RUN_MAIN:
    print("\n=== MAIN (std / CA / SNR-fixed / SNR-learn) ===")
    main_df = run(MAIN_DATASETS, ["std", "ca", "snr_fixed", "snr_learn"], MAIN_SEEDS)
    main_df.to_csv(os.path.join(OUTDIR, "ca_results_main.csv"), index=False)
else:
    main_df = EMPTY
if RUN_EXT:
    print("\n=== EXTENSIONS ===")
    ext_df = run(EXT_DATASETS, ["std", "ca", "snr_fixed", "snr_learn", "ll", "per_layer", "dynamic", "anisotropic"], EXT_SEEDS)
    ext_df.to_csv(os.path.join(OUTDIR, "ca_results_ext.csv"), index=False)
else:
    ext_df = EMPTY
RAN_DATASETS = (set(MAIN_DATASETS) if RUN_MAIN else set()) | (set(EXT_DATASETS) if RUN_EXT else set())

# ---------------------------------------------------------------------
#  Data-budget / noise sweep: std vs SNR-gated FNO (Sec. 7 predictions)
# ---------------------------------------------------------------------
def run_sweep():
    rows = []
    for ds in SWEEP_DATASETS:
        inc = IN_CH[ds]
        for N in SWEEP_N:
            for noise in SWEEP_NOISE:
                for seed in range(42, 42 + SWEEP_SEEDS):
                    set_seed(seed)
                    tr = get_dataset(ds, N, seed); te = get_dataset(ds, N_TEST, seed + 1000)
                    A = torch.stack([tr[i][0] for i in range(len(tr))])
                    U = torch.stack([tr[i][1] for i in range(len(tr))])
                    if noise > 0:   # observation noise on the TRAIN set only; test stays clean
                        A = A + noise * A.std() * torch.randn_like(A)
                        U = U + noise * U.std() * torch.randn_like(U)
                    um, ustd = U.mean().item(), U.std().item(); un = (U - um) / (ustd + 1e-8)
                    trl = DataLoader(TensorDataset(A, U), batch_size=BATCH, shuffle=True)
                    tel = DataLoader(te, batch_size=BATCH, shuffle=False)
                    nm = min(200, len(A)); h = nm // 2
                    Am, Um = A[:nm].cpu(), un[:nm].cpu()
                    r = {"Dataset": ds, "N": N, "noise": noise, "Seed": seed}
                    if any(mm in SWEEP_MODELS for mm in ["snr_fixed", "snr_learn"]):
                        g_low, g_high = compute_snr_prior(Am[:h], Um[:h], Am[h:], Um[h:], MODES, MODES)
                        r["gate_prior_mean"] = float((g_low.mean() + g_high.mean()) / 2)
                    if "std" in SWEEP_MODELS:
                        set_seed(seed); l, t, _, _, _ = train_model(FNO2d(inc), trl, tel, um, ustd); r["std_loss"] = l
                    if "snr_fixed" in SWEEP_MODELS:
                        set_seed(seed); m = SNRGatedFNO2d(g_low, g_high, inc, learnable=False)
                        l, t, m, _, _ = train_model(m, trl, tel, um, ustd)
                        r["snr_fixed_loss"] = l; r["snr_fixed_gate_final"] = m.mean_gate()
                    if "snr_learn" in SWEEP_MODELS:
                        set_seed(seed); m = SNRGatedFNO2d(g_low, g_high, inc, learnable=True)
                        l, t, m, _, _ = train_model(m, trl, tel, um, ustd)
                        r["snr_learn_loss"] = l; r["snr_learn_gate_final"] = m.mean_gate()
                    rows.append(r)
                    print(f"[sweep {ds} N={N} noise={noise} seed={seed}] "
                          + " ".join(f"{k.split('_')[0]}={r[k]:.4f}" for k in r if k.endswith('_loss')))
    return pd.DataFrame(rows)

if RUN_SWEEP:
    print("\n=== DATA-BUDGET / NOISE SWEEP ==="); sweep_df = run_sweep()
    sweep_df.to_csv(os.path.join(OUTDIR, "sweep_results.csv"), index=False)
else:
    sweep_df = EMPTY

# =====================================================================
#  LATEX TABLE ROWS
# =====================================================================
PRETTY={"poisson":"Poisson","heat":"Heat","advection":"Advection","darcy":"Darcy",
        "wave":"Wave","darcy_multi":"Darcy-multi","heat_sensor":"Heat-sensor"}
def _has(df, col, ds): return (not df.empty) and (col in df) and (not df[df.Dataset==ds][col].dropna().empty)
print("\n"+"="*70+"\nLATEX  — paste matching rows between the RESULTS_* markers\n"+"="*70)
print("% RESULTS_MAIN rows: Dataset & din & Std & CA(hard) & SNR-fixed & SNR-learn & SNRlearn-imp% & win% & p")
for ds in MAIN_DATASETS:
    if not _has(main_df,"std_loss",ds): continue
    s=main_df[main_df.Dataset==ds]; a=s["std_loss"].values
    def cell(c):
        return (f"{s[c].mean():.4f} $\\pm$ {s[c].std():.4f}" if _has(main_df,c,ds) else "--")
    # improvement / win / p reported for the headline SNR-learn variant vs std
    if _has(main_df,"snr_learn_loss",ds):
        b=s["snr_learn_loss"].values; imp=np.mean((a-b)/a)*100; win=np.mean(b<a)*100
        p=wilcoxon(a,b).pvalue if (wilcoxon and len(a)>1 and np.any(a!=b)) else float("nan")
        tail=f"{imp:+.1f} & {win:.0f} & {p:.3f}"
    else:
        tail="-- & -- & --"
    print(f"{PRETTY[ds]:<13s} & {IN_CH[ds]} & {cell('std_loss')} & {cell('ca_loss')} & "
          f"{cell('snr_fixed_loss')} & {cell('snr_learn_loss')} & {tail} \\\\")
print("\n% RESULTS_TIME rows (Std train / CA train / pre-analysis):")
for ds in ["heat","darcy_multi"]:
    if not _has(main_df,"std_time",ds): continue
    s=main_df[main_df.Dataset==ds]; e=ext_df[ext_df.Dataset==ds] if not ext_df.empty else EMPTY
    pre=e.ll_precomp_time.mean() if ("ll_precomp_time" in e and not e.empty) else float("nan")
    print(f"{PRETTY[ds]:<12s} & {s.std_time.mean():.2f} & {s.ca_time.mean():.2f} & {pre:.2f} \\\\")
print("\n% RESULTS_EXT rows (Std / CA / SNR-fixed / SNR-learn / +lambda / per-layer / dynamic):")
for ds in EXT_DATASETS:
    if ext_df.empty or ext_df[ext_df.Dataset==ds].empty: continue
    s=ext_df[ext_df.Dataset==ds]
    def g(c): return f"{s[c].mean():.4f}" if (c in s and not s[c].isna().all()) else "--"
    print(f"{PRETTY[ds]:<12s} & {g('std_loss')} & {g('ca_loss')} & {g('snr_fixed_loss')} & {g('snr_learn_loss')} & {g('ll_loss')} & {g('per_layer_loss')} & {g('dynamic_loss')} \\\\")

# SNR variants vs std on clean data (predicted: tie-or-better; report gates)
if not ext_df.empty and ("snr_fixed_loss" in ext_df or "snr_learn_loss" in ext_df):
    print("\n% SNR (fixed/learn) vs std on clean data [loss; gate prior->final; %imp; wins]:")
    for ds in EXT_DATASETS:
        s=ext_df[ext_df.Dataset==ds]
        if s.empty: continue
        a=s["std_loss"].values; prior=s.get("gate_prior_mean", pd.Series([float('nan')])).mean()
        parts=[f"std={a.mean():.4f}"]
        for v in ["snr_fixed","snr_learn"]:
            c=f"{v}_loss"
            if c not in s or s[c].dropna().empty: continue
            b=s[c].values; gf=s.get(f"{v}_gate_final", pd.Series([float('nan')])).mean()
            parts.append(f"{v}={b.mean():.4f} (g:{prior:.2f}->{gf:.2f}, {np.mean((a-b)/a)*100:+.1f}%, {int((b<a).sum())}/{len(a)})")
        print(f"  {ds:12s} "+"  ".join(parts))

# Sweep summary + figure (the headline shrinkage win)
SNR_COLS=[("snr_fixed_loss","SNR-fixed","C1"),("snr_learn_loss","SNR-learn","C2")]
if not sweep_df.empty:
    print("\n% SWEEP (improvement of each SNR variant over std, %):")
    for ds in sweep_df.Dataset.unique():
        for noise in sorted(sweep_df.noise.unique()):
            row=[]
            for N in sorted(sweep_df.N.unique()):
                s=sweep_df[(sweep_df.Dataset==ds)&(sweep_df.N==N)&(sweep_df.noise==noise)]
                if s.empty: continue
                cell=f"N={N}["
                for col,_,_ in SNR_COLS:
                    if col in s and not s[col].dropna().empty:
                        cell+=f"{col.split('_')[1][0]}:{(s.std_loss.values-s[col].values).mean()/s.std_loss.values.mean()*100:+.0f}% "
                row.append(cell.strip()+"]")
            print(f"  {ds:10s} noise={noise}:  "+"  ".join(row))
    ds_list=list(sweep_df.Dataset.unique()); noises=sorted(sweep_df.noise.unique())
    fig,axes=plt.subplots(len(ds_list),len(noises),figsize=(4*len(noises),3.2*len(ds_list)),squeeze=False)
    for r,ds in enumerate(ds_list):
        for c,noise in enumerate(noises):
            ax=axes[r][c]; sub=sweep_df[(sweep_df.Dataset==ds)&(sweep_df.noise==noise)]
            Ns=sorted(sub.N.unique())
            for col,lab,color in [("std_loss","Standard FNO","C0")]+SNR_COLS:
                if col not in sub or sub[col].dropna().empty: continue
                mu=[sub[sub.N==N][col].mean() for N in Ns]; sd=[sub[sub.N==N][col].std() for N in Ns]
                ax.errorbar(Ns,mu,yerr=sd,marker="o",label=lab,color=color,capsize=3)
            ax.set_title(f"{ds}, noise={noise}"); ax.set_xlabel("train size N"); ax.set_ylabel("test rel. $L_2$")
            ax.set_xscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,"figures","sweep.png"),dpi=130,bbox_inches="tight"); plt.close()

# =====================================================================
#  PER-MODE DIAGNOSTICS (kappa / energy / score / learned lambda)
# =====================================================================
def diagnostics(ds, seed=42):
    """Per-mode conditioning/energy/score and bilevel lambda over the low block.
    Saves a tidy CSV and a heatmap; returns the DataFrame."""
    set_seed(seed)
    d = get_dataset(ds, max(400, 2 * 100), seed)
    A = torch.stack([d[i][0] for i in range(len(d))])
    U = torch.stack([d[i][1] for i in range(len(d))])
    um, ustd = U.mean().item(), U.std().item(); un = (U - um) / (ustd + 1e-8)
    Am, Um = A[:200].cpu(), un[:200].cpu()
    af = torch.fft.rfft2(_to_cf(Am)); uf = torch.fft.rfft2(_to_cf(Um))
    _, _, Hf, Wf = af.shape; m2 = min(MODES, Wf)
    ll, hl, _, _ = compute_learnable_lambda(Am[:100], Um[:100], Am[100:], Um[100:], MODES, MODES)
    recs = []
    cond_map = np.zeros((MODES, m2)); en_map = np.zeros((MODES, m2))
    for i in range(min(MODES, Hf)):
        for j in range(m2):
            e, c = _energy_cond(af[:, :, i, j], uf[:, :, i, j])
            cond_map[i, j] = c; en_map[i, j] = e
            recs.append(dict(dataset=ds, ki=i, kj=j, energy=e, cond=c,
                             score=e / (c + 1e-8), lambda_learned=float(ll[i, j])))
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(OUTDIR, "diagnostics", f"{ds}_permode.csv"), index=False)
    # heatmaps
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a_, M, t in zip(ax, [np.log10(en_map + 1e-12), cond_map, np.log10(ll.numpy()[:, :m2] + 1e-9)],
                        ["log10 output energy", "condition number $\\kappa$", "log10 learned $\\lambda$"]):
        im = a_.imshow(M, aspect="auto", origin="upper"); a_.set_title(f"{ds}: {t}")
        a_.set_xlabel("$k_2$"); a_.set_ylabel("$k_1$"); fig.colorbar(im, ax=a_, fraction=0.046)
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "figures", f"diag_{ds}.png"), dpi=120, bbox_inches="tight"); plt.close()
    return df

if RUN_DIAG:
    print("\n=== PER-MODE DIAGNOSTICS ===")
    for ds in DIAG_DATASETS:
        if ds not in RAN_DATASETS: continue
        df = diagnostics(ds)
        hi = df.sort_values("energy", ascending=False).head(3)
        print(f"{ds}: top-energy modes  "
              + "; ".join(f"k=({r.ki},{r.kj}) E={r.energy:.2e} kappa={r.cond:.1f} lam={r.lambda_learned:.2f}"
                          for r in hi.itertuples()))

# =====================================================================
#  FIGURES  (only over datasets this notebook produced)
# =====================================================================
box_ds = [ds for ds in MAIN_DATASETS if not main_df.empty and not main_df[main_df.Dataset==ds].empty]
if box_ds:
    fig, axes = plt.subplots(1, len(box_ds), figsize=(3 * len(box_ds), 4), squeeze=False)
    for ax, ds in zip(axes[0], box_ds):
        s = main_df[main_df.Dataset == ds]
        ax.boxplot([s.std_loss.values, s.ca_loss.values], labels=["Std", "CA"], patch_artist=True)
        ax.set_title(ds); ax.set_yscale("log"); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("test rel. $L_2$")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "figures", "benchmark_boxplot.png"), dpi=130, bbox_inches="tight"); plt.close()

# convergence curves built from the per-epoch curves recorded this run
def _stack(ds, model):
    ks = [k for k in ALL_CURVES if k.startswith(f"{ds}|{model}|")]
    return np.array([ALL_CURVES[k]["test_relL2"] for k in ks]) if ks else None
cd = [ds for ds in ["heat", "wave", "heat_sensor"] if _stack(ds, "std") is not None and _stack(ds, "ca") is not None]
if cd:
    fig, axes = plt.subplots(1, len(cd), figsize=(5 * len(cd), 4), squeeze=False)
    for ax, ds in zip(axes[0], cd):
        ep = np.arange(1, EPOCHS + 1)
        for model, lab, col in [("std", "Standard FNO", "C0"), ("ca", "CA-FNO", "C1")]:
            arr = _stack(ds, model); m, s = arr.mean(0), arr.std(0)
            ax.plot(ep, m, col, label=lab); ax.fill_between(ep, m - s, m + s, alpha=0.2, color=col)
        ax.set_title(ds); ax.set_xlabel("epoch"); ax.set_ylabel("test rel. $L_2$")
        ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "figures", "convergence.png"), dpi=130, bbox_inches="tight"); plt.close()

# =====================================================================
#  SAVE EVERYTHING + BUNDLE
# =====================================================================
# per-epoch training/test curves for every trial
with open(os.path.join(OUTDIR, "curves", "all_curves.json"), "w") as fh:
    json.dump(ALL_CURVES, fh)
np.savez_compressed(os.path.join(OUTDIR, "curves", "all_curves.npz"),
                    **{k.replace("|", "__"): np.array([v["test_relL2"], v["train_mse"]])
                       for k, v in ALL_CURVES.items()})
# summary tables (aggregated)
def aggregate(df):
    out = []
    for ds in df.Dataset.unique():
        s = df[df.Dataset == ds]; row = {"Dataset": ds}
        for c in [c for c in s.columns if c.endswith("_loss")]:
            row[c + "_mean"] = s[c].mean(); row[c + "_std"] = s[c].std()
        out.append(row)
    return pd.DataFrame(out)
aggregate(main_df).to_csv(os.path.join(OUTDIR, "summary_main.csv"), index=False)
aggregate(ext_df).to_csv(os.path.join(OUTDIR, "summary_ext.csv"), index=False)
RUN_META["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
RUN_META["n_checkpoints"] = len(os.listdir(os.path.join(OUTDIR, "checkpoints")))
with open(os.path.join(OUTDIR, "metadata.json"), "w") as fh:
    json.dump(RUN_META, fh, indent=2)

# core bundle (small, fast): everything except the bulky checkpoints
CORE_ZIP, CKPT_ZIP = f"ca_artifacts_{RUN_LABEL}.zip", f"ca_checkpoints_{RUN_LABEL}.zip"
with zipfile.ZipFile(CORE_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, fs in os.walk(OUTDIR):
        if os.path.basename(root) == "checkpoints":
            continue
        for f in fs:
            p = os.path.join(root, f); z.write(p, os.path.relpath(p, "."))
# checkpoints bundle (optional, larger)
ckpt_dir = os.path.join(OUTDIR, "checkpoints")
ckpts = os.listdir(ckpt_dir)
if ckpts:
    with zipfile.ZipFile(CKPT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ckpts:
            p = os.path.join(ckpt_dir, f); z.write(p, os.path.relpath(p, "."))
print(f"\nSaved everything to ./{OUTDIR}/")
print("  contents:", {d: len(os.listdir(os.path.join(OUTDIR, d)))
                       for d in ["curves", "checkpoints", "diagnostics", "figures"]})
print(f"  {CORE_ZIP}: results, curves, diagnostics, figures, metadata (small)")
print(f"  {CKPT_ZIP}: model weights ({len(ckpts)} checkpoints)" if ckpts else "  (no checkpoints saved)")

try:
    from google.colab import files
    files.download(CORE_ZIP)
    if ckpts:
        files.download(CKPT_ZIP)
except Exception:
    pass
