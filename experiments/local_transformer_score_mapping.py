#%%
# transformer_gmm_score_local.py
# Drop-in replacement for your script with LOCAL (radius) attention: O(nk) instead of O(n^2).
import math, random, tqdm
from dataclasses import dataclass

import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#%%
# -------------------------
# Setup & hyperparameters
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
random.seed(0)

SEQ_LEN = 1000          # n
BATCH = 2              # sequences per step (keep small to fit memory)
STEPS = 8_000           # training steps
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

# Local attention neighborhood settings
RADIUS = 1.5
MAX_K  = 256
INCLUDE_SELF = True

# -------------------------
# Mixture utilities
# -------------------------
@dataclass
class GMM2D:
    weights: torch.Tensor        # (K,)
    means: torch.Tensor          # (K, 2)
    covs: torch.Tensor           # (K, 2, 2)
    precs: torch.Tensor          # (K, 2, 2)
    log_norm_consts: torch.Tensor # (K,)

def random_spd_2x2(batch_K: int) -> torch.Tensor:
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
    if K == 1:
        w = torch.ones(1)
    else:
        w = torch.rand(K)
        w = torch.clamp(w, MIN_MIX_WEIGHT, 1.0)
        w = w / w.sum()
    precs = torch.inverse(covs)
    # log((2π)^{d/2} |Σ|^{1/2}) with d=2  => log(2π) + 0.5 logdet(Σ)
    log_norm_consts = math.log(2 * math.pi) + 0.5 * torch.logdet(covs)
    return GMM2D(w, means, covs, precs, log_norm_consts)

def sample_from_gmm(gmm: GMM2D, n: int) -> torch.Tensor:
    K = gmm.weights.numel()
    comp = torch.multinomial(gmm.weights, num_samples=n, replacement=True)  # (n,)
    eps = torch.randn(n, 2)
    L = torch.linalg.cholesky(gmm.covs)  # (K,2,2)
    x = gmm.means[comp] + (L[comp] @ eps.unsqueeze(-1)).squeeze(-1)
    return x

def gmm_logpdf_components(x: torch.Tensor, gmm: GMM2D) -> torch.Tensor:
    """
    x: (B, N, 2)
    returns: (B, N, K)  log [ w_k * N_k(x) ]
    """
    B, N, _ = x.shape
    K = gmm.weights.numel()
    x_exp = x.unsqueeze(-2).expand(B, N, K, 2)      # (B,N,K,2)
    mean = gmm.means.view(1, 1, K, 2).to(x.dtype).to(x.device)
    prec = gmm.precs.view(1, 1, K, 2, 2).to(x.dtype).to(x.device)
    log_norm = gmm.log_norm_consts.view(1, 1, K).to(x.dtype).to(x.device)
    log_w = torch.log(gmm.weights.view(1, 1, K)).to(x.dtype).to(x.device)

    dx = x_exp - mean  # (B,N,K,2)
    qf = torch.einsum("...i,...ij,...j->...", dx, prec, dx)  # (B,N,K)
    log_N = -0.5 * qf - log_norm  # (B,N,K)
    return log_w + log_N

def gmm_score(x: torch.Tensor, gmm: GMM2D) -> torch.Tensor:
    """
    Exact score ∇ log f(x) for GMM f = Σ_k w_k N(·|μ_k, Σ_k)
    x: (B, N, 2)
    returns: (B, N, 2)
    """
    B, N, _ = x.shape
    K = gmm.weights.numel()
    log_wNk = gmm_logpdf_components(x, gmm)  # (B,N,K)
    log_f = torch.logsumexp(log_wNk, dim=-1, keepdim=True)  # (B,N,1)
    gamma = torch.softmax(log_wNk - log_f, dim=-1)          # (B,N,K)

    mean = gmm.means.view(1, 1, K, 2).to(x.dtype).to(x.device)
    prec = gmm.precs.view(1, 1, K, 2, 2).to(x.dtype).to(x.device)
    mu_minus_x = mean - x.unsqueeze(-2)                     # (B,N,K,2)
    comp_score = torch.einsum("...ij,...j->...i", prec, mu_minus_x)  # (B,N,K,2)
    weighted = (gamma.unsqueeze(-1) * comp_score).sum(dim=-2)        # (B,N,2)
    return weighted

# ------------------------------------------------------------
# Neighbor search: radius-limited, capped at max_k per token
# ------------------------------------------------------------
# @torch.no_grad()
# def radius_neighbors_2d(x: torch.Tensor, radius: float, max_k: int, include_self: bool = True):
#     """
#     Build neighbor indices per token using grid hashing (O(n) avg).
#     x:  (B, N, 2) on any device
#     returns:
#       nbr_idx  : (B, N, K) int64   (indices into dim=1)
#       nbr_mask : (B, N, K) bool    (True where a real neighbor exists)
#     """
#     device = x.device
#     B, N, _ = x.shape
#     K = min(max_k, max(1, N))

#     # Work on CPU for lightweight bucketing; move results back to device
#     x_cpu = x.detach().cpu()
#     all_idx = []
#     all_mask = []

#     r2 = radius * radius

#     for b in range(B):
#         xb = x_cpu[b]  # (N,2), float32
#         cells = torch.floor(xb / radius).to(torch.int64)  # (N,2)
#         cx, cy = cells[:, 0].tolist(), cells[:, 1].tolist()

#         # Buckets: (cx, cy) -> list of indices
#         buckets = {}
#         for i in range(N):
#             key = (cx[i], cy[i])
#             if key not in buckets:
#                 buckets[key] = []
#             buckets[key].append(i)

#         nbr_idx_b = torch.empty((N, K), dtype=torch.long)
#         nbr_msk_b = torch.zeros((N, K), dtype=torch.bool)

#         for i in range(N):
#             cell = (cx[i], cy[i])
#             cand = []
#             for dx in (-1, 0, 1):
#                 for dy in (-1, 0, 1):
#                     key = (cell[0] + dx, cell[1] + dy)
#                     if key in buckets:
#                         cand.extend(buckets[key])

#             if len(cand) > 64:
#                 cand = list(dict.fromkeys(cand))

#             pi = xb[i].unsqueeze(0)            # (1,2)
#             if len(cand) > 0:
#                 pc = xb[cand]                  # (M,2)
#                 dist2 = torch.sum((pc - pi) ** 2, dim=1)  # (M,)
#                 keep = (dist2 <= r2)
#                 cand = [cand[j] for j in torch.nonzero(keep).flatten().tolist()]

#             if include_self and (i not in cand):
#                 cand.append(i)

#             if len(cand) == 0:
#                 cand = [i]

#             if len(cand) > K:
#                 pc = xb[cand]
#                 dist2 = torch.sum((pc - pi) ** 2, dim=1)
#                 topk = torch.topk(-dist2, k=K, largest=True).indices
#                 cand = [cand[j] for j in topk.tolist()]

#             if len(cand) < K:
#                 pad = [cand[-1]] * (K - len(cand))
#                 full = cand + pad
#                 mask = [True] * len(cand) + [False] * (K - len(cand))
#             else:
#                 full = cand
#                 mask = [True] * K

#             nbr_idx_b[i] = torch.tensor(full, dtype=torch.long)
#             nbr_msk_b[i] = torch.tensor(mask, dtype=torch.bool)

#         all_idx.append(nbr_idx_b)
#         all_mask.append(nbr_msk_b)

#     nbr_idx = torch.stack(all_idx, dim=0).to(device)   # (B,N,K)
#     nbr_mask = torch.stack(all_mask, dim=0).to(device) # (B,N,K)
#     return nbr_idx, nbr_mask

@torch.no_grad()
def knn_radius_neighbors_gpu(x: torch.Tensor, max_k: int, radius: float | None = None, include_self: bool = True):
    """
    Fast GPU kNN with optional radius mask.

    x:        (B, N, 2)  on CUDA
    max_k:    number of neighbors to keep (including self if include_self=True)
    radius:   if not None, mask out neighbors with dist > radius
    returns:
      nbr_idx  : (B, N, K)  int64      indices into dim=1
      nbr_mask : (B, N, K)  bool       True where valid (passes radius)
    """
    device = x.device
    B, N, _ = x.shape
    K = min(max_k, N)

    # Pairwise distances (squared) on GPU: (B, N, N)
    # Note: cdist uses a fast GEMM-based kernel; with d=2 and N=1000 it's cheap.
    dist = torch.cdist(x, x, p=2)  # (B, N, N)
    dist2 = dist * dist

    # Control whether the diagonal (self) is considered
    if include_self:
        # Make sure self is the closest by zeroing diagonal
        eye = torch.eye(N, device=device).bool()
        dist2 = dist2.masked_fill(eye.unsqueeze(0), 0.0)
    else:
        big = torch.finfo(dist2.dtype).max
        eye = torch.eye(N, device=device).bool()
        dist2 = dist2.masked_fill(eye.unsqueeze(0), big)

    # Take K nearest neighbors (smallest distances)
    # We use negative to get smallest via topk(largest=True) which is faster on some backends.
    vals, idx = torch.topk(-dist2, k=K, dim=-1, largest=True)   # (B, N, K)
    nbr_idx = idx.to(torch.long)                                 # (B, N, K)

    # Radius mask (valid neighbors)
    if radius is None:
        nbr_mask = torch.ones((B, N, K), dtype=torch.bool, device=device)
    else:
        r2 = radius * radius
        # Gather the true distances for selected neighbors and compare to r^2
        sel_dist2 = torch.gather(dist2, dim=-1, index=nbr_idx)   # (B, N, K)
        nbr_mask = sel_dist2 <= r2

        # If we excluded self (include_self=False) and a row has <K valid nbrs,
        # we still keep indices but mark them invalid in the mask.

    return nbr_idx, nbr_mask


# ------------------------------------------------------------
# Local Multi-Head Self-Attention over neighbors
# ------------------------------------------------------------
class LocalMHSA(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.h = nhead
        self.dh = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.dh)

    def forward(self, x, nbr_idx, nbr_mask):
        """
        x:        (B,N,d)
        nbr_idx:  (B,N,K) int64
        nbr_mask: (B,N,K) bool  (True where valid neighbor)
        returns:  (B,N,d)
        """
        B, N, D = x.shape
        H, Dh = self.h, self.dh
        K = nbr_idx.shape[2]

        qkv = self.qkv(x)  # (B,N,3D)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        q = q.view(B, N, H, Dh)
        k = k.view(B, N, H, Dh)
        v = v.view(B, N, H, Dh)

        # heads-first to gather along sequence dim
        k_th = k.permute(0, 2, 1, 3).contiguous()   # (B,H,N,Dh)
        v_th = v.permute(0, 2, 1, 3).contiguous()   # (B,H,N,Dh)

        # flatten heads
        k_flat = k_th.reshape(B * H, N, Dh)         # (B*H,N,Dh)
        v_flat = v_th.reshape(B * H, N, Dh)         # (B*H,N,Dh)

        # neighbor indices per head
        idx_bhn = nbr_idx.unsqueeze(1).expand(B, H, N, K).reshape(B * H, N, K)  # (B*H,N,K)

        # expand input to 4D so gather has matching dims
        idx_exp = idx_bhn.unsqueeze(-1).expand(B * H, N, K, Dh)                 # (B*H,N,K,Dh)
        k_exp = k_flat.unsqueeze(2).expand(B * H, N, K, Dh)                      # (B*H,N,K,Dh)
        v_exp = v_flat.unsqueeze(2).expand(B * H, N, K, Dh)                      # (B*H,N,K,Dh)

        # gather neighbors along sequence dimension
        k_nb = torch.gather(k_exp, dim=1, index=idx_exp)                         # (B*H,N,K,Dh)
        v_nb = torch.gather(v_exp, dim=1, index=idx_exp)                         # (B*H,N,K,Dh)

        # reshape back to (B,N,K,H,Dh)
        k_nb = k_nb.view(B, H, N, K, Dh).permute(0, 2, 3, 1, 4).contiguous()
        v_nb = v_nb.view(B, H, N, K, Dh).permute(0, 2, 3, 1, 4).contiguous()

        # attention
        scores = (q.unsqueeze(2) * k_nb).sum(dim=-1) * self.scale               # (B,N,K,H)
        if nbr_mask is not None:
            scores = scores.masked_fill((~nbr_mask).unsqueeze(-1), float('-inf'))
        attn = F.softmax(scores, dim=2)                                         # (B,N,K,H)
        attn = self.drop(attn)

        out_h = (attn.unsqueeze(-1) * v_nb).sum(dim=2)                           # (B,N,H,Dh)
        out = out_h.reshape(B, N, D)
        out = self.proj(out)
        return out


class LocalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = LocalMHSA(d_model, nhead, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, nbr_idx, nbr_mask):
        x = x + self.attn(self.norm1(x), nbr_idx, nbr_mask)
        x = x + self.ff(self.norm2(x))
        return x

class LocalScoreTransformer(nn.Module):
    """
    Same API as your ScoreTransformer, but requires neighbor indices/masks.
    """
    def __init__(self, d_model=128, nhead=8, num_layers=4, ff_dim=256, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(2, d_model)
        self.blocks = nn.ModuleList([
            LocalTransformerBlock(d_model, nhead, ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.out_head = nn.Linear(d_model, 2)

    def forward(self, x, nbr_idx, nbr_mask):
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, nbr_idx, nbr_mask)
        y = self.out_head(h)
        return y

# -------------------------
# Training loop
# -------------------------
model = LocalScoreTransformer(D_MODEL, NHEAD, LAYERS, FF_DIM).to(device)
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

    # build neighbors for this batch
    nbr_idx, nbr_mask = knn_radius_neighbors_gpu(X, radius=RADIUS, max_k=MAX_K, include_self=INCLUDE_SELF)

    opt.zero_grad(set_to_none=True)
    pred = model(X, nbr_idx, nbr_mask)
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
    nbr_idx, nbr_mask = knn_radius_neighbors_gpu(X, radius=RADIUS, max_k=MAX_K, include_self=INCLUDE_SELF)
    S_pred = model(X, nbr_idx, nbr_mask)
    mse = F.mse_loss(S_pred, S_true).item()
    print(f"Eval MSE on a fresh K=2 mixture: {mse:.6f}")

#%%
# --- Visualization (uses local attention too) ---
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
    nbr_idx, nbr_mask = knn_radius_neighbors_gpu(X, radius=RADIUS, max_k=MAX_K, include_self=INCLUDE_SELF)
    S_pred = model(X, nbr_idx, nbr_mask)                # (1,n,2)

    x_np = X[0].cpu().numpy()
    t_np = (-0.5 * S_true[0]).cpu().numpy()
    p_np = (-0.5 * S_pred[0]).cpu().numpy()

    idx = torch.randperm(n)[:m].cpu().numpy()
    pts  = x_np[idx]
    tru  = t_np[idx]
    prd  = p_np[idx]

    plt.figure(figsize=(6, 6))
    plt.scatter(x_np[:, 0], x_np[:, 1], s=5, alpha=0.3, label='data points')

    plt.quiver(
        pts[:, 0], pts[:, 1],
        tru[:, 0], tru[:, 1],
        angles='xy', scale_units='xy', scale=1, width=0.004, alpha=0.9,
        label='true: -½∇log f',
        color='orange'
    )

    plt.quiver(
        pts[:, 0], pts[:, 1],
        prd[:, 0], prd[:, 1],
        angles='xy', scale_units='xy', scale=1, width=0.0025, alpha=0.9,
        label=f'pred: -½s, mse {F.mse_loss(S_pred, S_true).item():.4f}',
        color='blue'
    )

    plt.axis('equal')
    plt.title(f'Local-Attention Transformer (radius={RADIUS}, K≤{MAX_K})')
    plt.legend(loc='upper right')
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=200)
    plt.show()

#%%
# Example usage after training:
visualize_scores(model, n=1000, m=20, K=1)
visualize_scores(model, n=1000, m=20, K=2)
visualize_scores(model, n=1000, m=20, K=2)
visualize_scores(model, n=1000, m=20, K=2)

#%%

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _ensure_eval(m):
    if m is None:
        raise RuntimeError("model is not defined.")
    m.eval()
    return m

def _bench_local_transformer(mdl, X_max, ns, warmup=20, repeats=5):
    mdl = _ensure_eval(mdl)
    param_dtype = next(mdl.parameters()).dtype
    times_ms = []

    # light global warmup
    with torch.inference_mode():
        for _ in range(warmup):
            n0 = int(ns[0])
            Xw = X_max[:, :n0, :].to(param_dtype)
            nbr_idx, nbr_mask = knn_radius_neighbors_gpu(
                Xw, max_k=MAX_K, radius=RADIUS, include_self=INCLUDE_SELF
            )
            _ = mdl(Xw, nbr_idx, nbr_mask)
        _sync()

    # per-size timing
    for n in tqdm.tqdm(ns):
        Xn = X_max[:, :n, :].to(param_dtype)

        # size-specific warmup
        with torch.inference_mode():
            for _ in range(warmup):
                nbr_idx, nbr_mask = knn_radius_neighbors_gpu(
                    Xn, max_k=MAX_K, radius=RADIUS, include_self=INCLUDE_SELF
                )
                _ = mdl(Xn, nbr_idx, nbr_mask)
            _sync()

            t0 = time.perf_counter()
            for _ in range(repeats):
                nbr_idx, nbr_mask = knn_radius_neighbors_gpu(
                    Xn, max_k=MAX_K, radius=RADIUS, include_self=INCLUDE_SELF
                )
                _ = mdl(Xn, nbr_idx, nbr_mask)
            _sync()
            t1 = time.perf_counter()

        times_ms.append(1000.0 * (t1 - t0) / repeats)

    return times_ms

# prepare inputs once (max size)
ns = [2 ** k for k in range(4, 17)]
Nmax = ns[-1]
X_max = torch.randn(1, Nmax, 2, device=device)

model.eval()
warmup = 50
times_local = _bench_local_transformer(model, X_max, ns, warmup=warmup, repeats=5)

# Plot
plt.figure(figsize=(7, 5))
plt.plot(ns, times_local, "-o", label="Local Transformer (end-to-end)")
plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("n (number of points)")
plt.ylabel("Time (ms, log scale)")
plt.title("Time vs number of points (Local Attention, incl. neighbor build)")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

#%%
# Timing table (console + matplotlib)
headers = ["n", "Local Transformer (ms)"]
rows_num = list(zip(ns, times_local))
rows_str = [[f"{n}", f"{t0:.3f}"] for n, t0 in rows_num]

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
plt.figure(figsize=(6, 0.5 + 0.3 * len(rows_str)))
plt.axis("off")
tbl = plt.table(cellText=rows_str, colLabels=headers, loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.1, 1.2)
plt.title("Timing summary (ms)")
plt.tight_layout()
plt.show()