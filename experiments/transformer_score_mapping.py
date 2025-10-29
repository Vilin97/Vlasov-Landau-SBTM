#%%
# transformer_gmm_score.py
import math
import random
from dataclasses import dataclass
from typing import Tuple

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import time
import copy

#%%
# -------------------------
# Config
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
random.seed(0)

SEQ_LEN = 1000          # n
BATCH = 32               # sequences per step (keep small to fit memory)
STEPS = 2_000          # training steps
LR = 2e-4
PRINT_EVERY = 200
D_MODEL = 128
NHEAD = 8
LAYERS = 4
FF_DIM = 256
WEIGHT_DECAY = 1e-6

MEAN_RANGE = 4.0        # means sampled from [-MEAN_RANGE, MEAN_RANGE]^2
EIG_MIN, EIG_MAX = 0.3, 2.0  # eigenvalues for SPD Σ
MIN_MIX_WEIGHT = 0.2    # to avoid degenerate mixtures
K_CHOICES = [1, 2]      # number of components per step

# -------------------------
# Mixture utilities
# -------------------------
@dataclass
class GMM2D:
    # K components
    weights: torch.Tensor        # (K,)
    means: torch.Tensor          # (K, 2)
    covs: torch.Tensor           # (K, 2, 2)
    precs: torch.Tensor          # (K, 2, 2)  (Σ^{-1})
    log_norm_consts: torch.Tensor # (K,)  log normalizer for mvn pdf (no weight)

def random_spd_2x2(batch_K: int) -> torch.Tensor:
    # random eigenvalues in [EIG_MIN, EIG_MAX], random rotation
    theta = torch.rand(batch_K) * 2 * math.pi
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.stack([torch.stack([c, -s], dim=-1),
                     torch.stack([s,  c], dim=-1)], dim=-2)  # (K,2,2)
    eig = EIG_MIN + (EIG_MAX - EIG_MIN) * torch.rand(batch_K, 2)
    D = torch.zeros(batch_K, 2, 2)
    D[:, 0, 0], D[:, 1, 1] = eig[:, 0], eig[:, 1]
    cov = R @ D @ R.transpose(-1, -2)
    return cov

def sample_gmm2d(K: int) -> GMM2D:
    means = (torch.rand(K, 2) * 2 * MEAN_RANGE - MEAN_RANGE)
    covs = random_spd_2x2(K)
    # normalize random weights, enforce min weight if K=2
    if K == 1:
        w = torch.ones(1)
    else:
        w = torch.rand(K)
        w = torch.clamp(w, MIN_MIX_WEIGHT, 1.0)
        w = w / w.sum()
    # precompute precisions and log-normalizers
    precs = torch.inverse(covs)
    # log((2π)^{d/2} |Σ|^{1/2}) = d/2 log(2π) + 1/2 logdet(Σ), d=2
    log_norm_consts = math.log(2 * math.pi) + 0.5 * torch.logdet(covs)
    return GMM2D(w, means, covs, precs, log_norm_consts)

def sample_from_gmm(gmm: GMM2D, n: int) -> torch.Tensor:
    K = gmm.weights.numel()
    comp = torch.multinomial(gmm.weights, num_samples=n, replacement=True)  # (n,)
    eps = torch.randn(n, 2)
    # draw via Cholesky
    L = torch.linalg.cholesky(gmm.covs)  # (K,2,2)
    x = gmm.means[comp] + (L[comp] @ eps.unsqueeze(-1)).squeeze(-1)
    return x

def log_gauss_unnorm(x: torch.Tensor, mean: torch.Tensor, prec: torch.Tensor) -> torch.Tensor:
    # returns log N(x|mean, Σ) + const (missing the -(d/2)log(2π) - 1/2 logdet Σ part)
    # We’ll subtract the exact log normalizer separately.
    dx = (x - mean)  # (...,2)
    # quadratic form: (x-μ)^T Σ^{-1} (x-μ)
    qf = torch.einsum("...i,ij,...j->...", dx, prec, dx)
    return -0.5 * qf  # (...,)

def gmm_logpdf_components(x: torch.Tensor, gmm: GMM2D) -> torch.Tensor:
    """
    x: (B, N, 2)
    returns: (B, N, K)  log [ w_k * N_k(x) ]
    """
    B, N, _ = x.shape
    K = gmm.weights.numel()
    # expand to broadcast
    x_exp = x.unsqueeze(-2).expand(B, N, K, 2)      # (B,N,K,2)
    mean = gmm.means.view(1, 1, K, 2)
    prec = gmm.precs.view(1, 1, K, 2, 2)
    log_norm = gmm.log_norm_consts.view(1, 1, K)
    log_w = torch.log(gmm.weights.view(1, 1, K))

    # log N part
    dx = x_exp - mean  # (B,N,K,2)
    qf = torch.einsum("...i,...ij,...j->... ", dx, prec, dx)  # (B,N,K)
    log_N = -0.5 * qf - log_norm  # (B,N,K)
    return log_w + log_N          # (B,N,K)

def gmm_score(x: torch.Tensor, gmm: GMM2D) -> torch.Tensor:
    """
    Exact score ∇ log f(x) for GMM f = Σ_k w_k N(·|μ_k, Σ_k)
    x: (B, N, 2)
    returns: (B, N, 2)
    Formula: ∇ log f(x) = Σ_k γ_k(x) Σ_k^{-1}(μ_k - x),  γ_k = w_k N_k / f
    """
    B, N, _ = x.shape
    K = gmm.weights.numel()
    log_wNk = gmm_logpdf_components(x, gmm)  # (B,N,K)
    log_f = torch.logsumexp(log_wNk, dim=-1, keepdim=True)  # (B,N,1)
    gamma = torch.softmax(log_wNk - log_f, dim=-1)          # (B,N,K)

    # Σ_k^{-1}(μ_k - x)
    mean = gmm.means.view(1, 1, K, 2)
    prec = gmm.precs.view(1, 1, K, 2, 2)
    mu_minus_x = mean - x.unsqueeze(-2)                     # (B,N,K,2)
    comp_score = torch.einsum("...ij,...j->...i", prec, mu_minus_x)  # (B,N,K,2)
    weighted = (gamma.unsqueeze(-1) * comp_score).sum(dim=-2)        # (B,N,2)
    return weighted

# -------------------------
# Transformer model
# -------------------------
class ScoreTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4, ff_dim=256):
        super().__init__()
        self.in_proj = nn.Linear(2, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_head = nn.Linear(d_model, 2)
        # no positional encodings -> permutation-equivariant mapping

    def forward(self, x):  # x: (B, N, 2)
        h = self.in_proj(x)
        h = self.encoder(h)   # (B, N, d_model)
        y = self.out_head(h)  # (B, N, 2)
        return y

# -------------------------
# Training loop
# -------------------------
model = ScoreTransformer(D_MODEL, NHEAD, LAYERS, FF_DIM).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def one_step():
    # sample a fresh GMM and a batch of sequences
    Ks = [random.choice(K_CHOICES) for _ in range(BATCH)]
    gmms = [sample_gmm2d(K) for K in Ks]

    # pack into tensors
    X_list, S_list = [], []
    for gmm in gmms:
        x = sample_from_gmm(gmm, SEQ_LEN).to(device)              # (N,2)
        xB = x.unsqueeze(0)                                       # (1,N,2)
        s = gmm_score(xB, GMM2D(gmm.weights.to(device),
                                gmm.means.to(device),
                                gmm.covs.to(device),
                                gmm.precs.to(device),
                                gmm.log_norm_consts.to(device)))  # (1,N,2)
        X_list.append(xB)
        S_list.append(s)
    X = torch.cat(X_list, dim=0)  # (B,N,2)
    S = torch.cat(S_list, dim=0)  # (B,N,2)

    opt.zero_grad(set_to_none=True)
    pred = model(X)
    loss = F.mse_loss(pred, S)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()

#%%
ema = None
beta = 0.98
for step in tqdm.tqdm(range(1, STEPS + 1)):
    loss = one_step()
    ema = loss if ema is None else beta * ema + (1 - beta) * loss
    if step % PRINT_EVERY == 0:
        print(f"step {step:5d}  loss {loss:.6f}  ema {ema:.6f}")

# Quick sanity check on a fixed mixture
with torch.no_grad():
    gmm = sample_gmm2d(2)
    X = sample_from_gmm(gmm, SEQ_LEN).unsqueeze(0).to(device)  # (1,N,2)
    S_true = gmm_score(
        X, GMM2D(gmm.weights.to(device), gmm.means.to(device),
                    gmm.covs.to(device), gmm.precs.to(device), gmm.log_norm_consts.to(device))
    )
    S_pred = model(X)
    mse = F.mse_loss(S_pred, S_true).item()
    print(f"Eval MSE on a fresh K=2 mixture: {mse:.6f}")

#%%
# --- add to transformer_gmm_score.py ---
import matplotlib.pyplot as plt

@torch.no_grad()
def visualize_scores(model, n=1000, m=20, K=None, fname=None):
    """
    Sample a GMM (optionally fix K in {1,2}), draw n points,
    and plot -0.5 * (true score) and -0.5 * (pred score) for m points.
    """
    model.eval()
    K = int(K) if K in (1, 2) else random.choice([1, 2])
    gmm = sample_gmm2d(K)
    print("GMM parameters:")
    print(f"  K = {gmm.weights.numel()}")
    print(f"  weights:\n{gmm.weights}")
    print(f"  means:\n{gmm.means}")
    print(f"  covariances:\n{gmm.covs}")

    X = sample_from_gmm(gmm, n).unsqueeze(0).to(device)  # (1,n,2)
    S_true = gmm_score(
        X, GMM2D(gmm.weights.to(device), gmm.means.to(device),
                 gmm.covs.to(device), gmm.precs.to(device), gmm.log_norm_consts.to(device))
    )  # (1,n,2)
    S_pred = model(X)                                    # (1,n,2)

    x_np = X[0].cpu().numpy()
    t_np = (-0.5 * S_true[0]).cpu().numpy()
    p_np = (-0.5 * S_pred[0]).cpu().numpy()

    # pick a small subset of points for arrows
    idx = torch.randperm(n)[:m].cpu().numpy()
    pts  = x_np[idx]
    tru  = t_np[idx]
    prd  = p_np[idx]

    # scatter all points faintly; arrows for the subset
    plt.figure(figsize=(6, 6))
    plt.scatter(x_np[:, 0], x_np[:, 1], s=5, alpha=0.3, label='data points')

    # true score arrows (solid)
    plt.quiver(
        pts[:, 0], pts[:, 1],
        tru[:, 0], tru[:, 1],
        angles='xy', scale_units='xy', scale=1, width=0.004, alpha=0.9,
        label='true: -½∇log f',
        color='orange'
    )

    # predicted score arrows (dashed effect via shorter head + overplot)
    plt.quiver(
        pts[:, 0], pts[:, 1],
        prd[:, 0], prd[:, 1],
        angles='xy', scale_units='xy', scale=1, width=0.0025, alpha=0.9,
        label=f'pred: -½s, mse {F.mse_loss(S_pred, S_true).item():.4f}',
        color='blue'
    )

    plt.axis('equal')
    plt.title(f'Universal Score Prediction with a Transformer')
    plt.legend(loc='upper right')
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=200)
    plt.show()

# Example usage after training:
visualize_scores(model, n=1000, m=20, K=1)
visualize_scores(model, n=1000, m=20, K=2)
visualize_scores(model, n=1000, m=20, K=2)
visualize_scores(model, n=1000, m=20, K=2)

#%%
@torch.no_grad()
def sample_batch_from_gmm(n=1000, K=None):
    # K = int(K) if K in (1, 2) else random.choice([1, 2])
    gmm = sample_gmm2d(K)
    X = sample_from_gmm(gmm, n).unsqueeze(0).to(device)  # (1,n,2)
    S_true = gmm_score(
        X, GMM2D(gmm.weights.to(device), gmm.means.to(device),
                 gmm.covs.to(device), gmm.precs.to(device), gmm.log_norm_consts.to(device))
    )  # (1,n,2)
    return gmm, X, S_true

@torch.no_grad()
def plot_scores(model, X, gmm, arrow_idx, title_prefix=""):
    """
    Plot -0.5 * score arrows for m points: true (solid) vs model (dashed).
    X: (1,n,2) on device
    """
    model.eval()
    S_true = gmm_score(
        X, GMM2D(gmm.weights.to(device), gmm.means.to(device),
                 gmm.covs.to(device), gmm.precs.to(device), gmm.log_norm_consts.to(device))
    )  # (1,n,2)
    S_pred = model(X)  # (1,n,2)

    x_np = X[0].cpu().numpy()
    tru  = (-0.5 * S_true[0]).cpu().numpy()
    prd  = (-0.5 * S_pred[0]).cpu().numpy()

    # subset for arrows
    pts, tru_m, prd_m = x_np[arrow_idx], tru[arrow_idx], prd[arrow_idx]

    mse = F.mse_loss(S_pred, S_true).item()

    plt.figure(figsize=(6, 6))
    plt.scatter(x_np[:, 0], x_np[:, 1], s=5, alpha=0.3, label='data points')

    plt.quiver(
        pts[:, 0], pts[:, 1],
        tru_m[:, 0], tru_m[:, 1],
        angles='xy', scale_units='xy', scale=1, width=0.004, alpha=0.9,
        label='true: -½∇log f',
        color='orange'
    )
    plt.quiver(
        pts[:, 0], pts[:, 1],
        prd_m[:, 0], prd_m[:, 1],
        angles='xy', scale_units='xy', scale=1, width=0.0025, alpha=0.9,
        label=f'pred: -½s, mse {mse:.4f}',
        color='blue'
    )

    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.title(f"{title_prefix} true vs predicted score")
    plt.tight_layout()
    plt.show()

# -------- MLP model --------
class ScoreMLP(nn.Module):
    def __init__(self, width=128, depth=4, in_dim=2, out_dim=2):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # x: (B,N,2) or (N,2)
        y = self.net(x if x.dim()==2 else x.view(-1, x.shape[-1]))
        return y.view(x.shape[:-1] + (2,))

# -------- implicit score-matching loss (PyTorch) --------
def loss_implicit_pt(model, v, alpha=0.05):
    """
    v: (1,n,2) or (n,2) — we use (n,2) inside.
    Implements E[ ||s(x)||^2 + z·(s(x+αz)-s(x-αz))/α ], with z ~ N(0,I).
    """
    v = v.view(-1, 2)
    n = v.shape[0]
    z = torch.randn_like(v)  # (n,2)
    s_pred = model(v)                    # (n,2)
    s_lo   = model(v - alpha * z)        # (n,2)
    s_hi   = model(v + alpha * z)        # (n,2)
    div_s  = torch.sum((s_hi - s_lo) * z) / alpha  # scalar
    loss   = (torch.sum(s_pred ** 2) + div_s) / n
    return loss

# -------- train MLP on ONE fixed sample cloud (non-universal setting) --------
def train_mlp_on_fixed_cloud(X, steps=5000, lr=1e-3, alpha=0.05, wd=0.0, verbose_every=250):
    """
    X: (1,n,2) tensor on device (fixed cloud sampled once).
    Optimizes implicit loss on that fixed empirical distribution.
    """
    mlp = ScoreMLP(width=128, depth=4).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=wd)
    ema = None; beta = 0.98
    for t in tqdm.tqdm(range(1, steps+1)):
        opt.zero_grad(set_to_none=True)
        loss = loss_implicit_pt(mlp, X[0], alpha=alpha)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        opt.step()
        ema = loss.item() if ema is None else beta*ema + (1-beta)*loss.item()
        if t % verbose_every == 0:
            print(f"[MLP] step {t:5d}  loss {loss.item():.6f}  ema {ema:.6f}")
    return mlp

#%%
# 1) Fix one mixture and one point cloud
gmm_fixed, X_fixed, S_fixed = sample_batch_from_gmm(n=1000, K=2)
arrow_idx = torch.randperm(X_fixed.shape[1])[:30].cpu().numpy()

# 2) Train the MLP on THIS fixed cloud with implicit loss
steps = 10
mlp = train_mlp_on_fixed_cloud(X_fixed, steps=steps, lr=1e-3, alpha=0.05)

# 3) Visualize MLP vs truth on the fixed cloud
plot_scores(mlp, X_fixed, gmm_fixed, arrow_idx, title_prefix=f"MLP (implicit SM, {steps} steps) | ")

# 2) Train the MLP on THIS fixed cloud with implicit loss
steps = 100
mlp = train_mlp_on_fixed_cloud(X_fixed, steps=steps, lr=1e-3, alpha=0.05)

# 3) Visualize MLP vs truth on the fixed cloud
plot_scores(mlp, X_fixed, gmm_fixed, arrow_idx, title_prefix=f"MLP (implicit SM, {steps} steps) | ")

steps = 1000
mlp = train_mlp_on_fixed_cloud(X_fixed, steps=steps, lr=1e-3, alpha=0.05, verbose_every=10000)

# 3) Visualize MLP vs truth on the fixed cloud
plot_scores(mlp, X_fixed, gmm_fixed, arrow_idx, title_prefix=f"MLP (implicit SM, {steps} steps) | ")

# 4) Visualize Transformer vs truth on the fixed cloud
plot_scores(model, X_fixed, gmm_fixed, arrow_idx, title_prefix="Transformer (universal) | ")

#%%
gmm_fixed, X_fixed, S_fixed = sample_batch_from_gmm(n=100000, K=5)
arrow_idx = torch.randperm(X_fixed.shape[1])[:30].cpu().numpy()
plot_scores(model, X_fixed, gmm_fixed, arrow_idx, title_prefix="Transformer (universal) | ")

gmm_fixed, X_fixed, S_fixed = sample_batch_from_gmm(n=100000, K=5)
arrow_idx = torch.randperm(X_fixed.shape[1])[:30].cpu().numpy()
plot_scores(model, X_fixed, gmm_fixed, arrow_idx, title_prefix="Transformer (universal) | ")

#%%
# Benchmark inference time vs number of points for Transformer (model) and MLP (mlp)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _ensure_model(name, mdl):
    if mdl is None:
        raise RuntimeError(f"{name} is not defined.")
    mdl.eval()
    return mdl

def _bench_model(mdl, X_max, ns, warmup=5, repeats=5):
    mdl = _ensure_model("model", mdl)
    dtype = next(mdl.parameters()).dtype
    times_ms = []
    # light global warmup
    with torch.inference_mode():
        for _ in range(warmup):
            _ = mdl(X_max[:, :min(ns), :].to(dtype))
        _sync()
    # per-size timing
    for n in tqdm.tqdm(ns):
        Xn = X_max[:, :n, :].to(dtype)
        # size-specific warmup
        with torch.inference_mode():
            for _ in range(warmup):
                _ = mdl(Xn)
            _sync()
            t0 = time.perf_counter()
            for _ in range(repeats):
                _ = mdl(Xn)
            _sync()
            t1 = time.perf_counter()
        times_ms.append(1000.0 * (t1 - t0) / repeats)
    return times_ms

def _bench_mlp_gd_step(mlp_ref, X_max, ns, lr=1e-3, alpha=0.05, wd=0.0, clip=1.0, warmup=3, repeats=3):
    if mlp_ref is None:
        raise RuntimeError("mlp is not defined.")
    times_ms = []
    param_dtype = next(mlp_ref.parameters()).dtype
    for n in tqdm.tqdm(ns):
        # fresh copy so weight updates don't affect subsequent sizes
        mlp_tmp = copy.deepcopy(mlp_ref).to(device).train()
        opt = torch.optim.AdamW(mlp_tmp.parameters(), lr=lr, weight_decay=wd)
        v = X_max[0, :n, :].to(param_dtype)  # (n,2)

        # warmup steps (unmeasured)
        for _ in range(warmup):
            opt.zero_grad(set_to_none=True)
            loss = loss_implicit_pt(mlp_tmp, v, alpha=alpha)
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(mlp_tmp.parameters(), clip)
            opt.step()
        _sync()

        # timed repeats
        t0 = time.perf_counter()
        for _ in range(repeats):
            opt.zero_grad(set_to_none=True)
            loss = loss_implicit_pt(mlp_tmp, v, alpha=alpha)
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(mlp_tmp.parameters(), clip)
            opt.step()
        _sync()
        t1 = time.perf_counter()
        times_ms.append(1000.0 * (t1 - t0) / repeats)
        del mlp_tmp, opt
    return times_ms

# prepare inputs once (max size)
ns = [2 ** k for k in range(4, 17)]
Nmax = ns[-1]
X_max = torch.randn(1, Nmax, 2, device=device)

# ensure mlp exists; if not, create an untrained one for timing
if "mlp" not in globals():
    mlp = ScoreMLP().to(device).eval()

model.eval()
mlp.eval()

warmup = 50
times_model    = _bench_model(model, X_max, ns, warmup=warmup, repeats=5)
times_mlp      = _bench_model(mlp,   X_max, ns, warmup=warmup, repeats=5)
times_mlp_gd   = _bench_mlp_gd_step(mlp, X_max, ns, lr=1e-3, alpha=0.05, warmup=warmup, repeats=5)

# Plot
plt.figure(figsize=(7, 5))
plt.plot(ns, times_model, "-o", label="Transformer (model) inference")
plt.plot(ns, times_mlp, "-o", label="MLP (mlp) inference")
plt.plot(ns, times_mlp_gd, "-o", label="MLP GD step (implicit loss)")
plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("n (number of points)")
plt.ylabel("Time (ms, log scale)")
plt.title("Time vs number of points")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

#%%
# Build and display a timing table (console + matplotlib)
headers = ["n", "Transformer (ms)", "MLP (ms)", "MLP GD step (ms)"]
rows_num = list(zip(ns, times_model, times_mlp, times_mlp_gd))
rows_str = [[f"{n}", f"{t0:.3f}", f"{t1:.3f}", f"{t2:.3f}"] for n, t0, t1, t2 in rows_num]

# Console table
col_widths = [max(len(h), max((len(r[i]) for r in rows_str), default=0)) for i, h in enumerate(headers)]
sep = "+".join("-" * (w + 2) for w in col_widths)
print(sep.replace("-", "-"))
print("| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |")
print(sep.replace("-", "-"))
for r in rows_str:
    print("| " + " | ".join(r[i].rjust(col_widths[i]) for i in range(len(headers))) + " |")
print(sep.replace("-", "-"))

# Matplotlib table
plt.figure(figsize=(8, 0.5 + 0.3 * len(rows_str)))
plt.axis("off")
tbl = plt.table(cellText=rows_str, colLabels=headers, loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.1, 1.2)
plt.title("Timing summary (ms)")
plt.tight_layout()
plt.show()

#%%
X = torch.randn(1, 2**18, 2, device=device) # 260,000 points
t = time.perf_counter()
model.eval()
model(X) # takes about 10 Gb of GPU memory
torch.cuda.synchronize()
t2 = time.perf_counter()
print(f"Transformer inference on 2^18 points took {(t2 - t)*1000:.2f} ms") # about 100 seconds