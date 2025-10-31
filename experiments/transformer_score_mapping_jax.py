# %%
# transformer_gmm_score_jax.py
import math
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random, jit
from flax import nnx
import optax
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#%%
# -------------------------
# Config
# -------------------------
SEED          = 0
key_master    = random.PRNGKey(SEED)

SEQ_LEN       = 1000          # n
BATCH         = 4            # sequences per step
STEPS         = 50
LR            = 2e-4
PRINT_EVERY   = 200
WEIGHT_DECAY  = 1e-6

D_MODEL       = 128
NHEAD         = 8
LAYERS        = 4
FF_DIM        = 256

DIM           = 2             # <- set any d >= 1
MEAN_RANGE    = 4.0
EIG_MIN       = 0.3
EIG_MAX       = 2.0
MIN_MIX_WEIGHT= 0.2
MAX_K         = 2             # we pad to 2 comps for vectorization
K_CHOICES     = jnp.array([1, 2])

# -------------------------
# Mixture utilities (general d)
# -------------------------
@dataclass
class GMM:
    weights: jnp.ndarray        # (B, K)
    means:   jnp.ndarray        # (B, K, d)
    covs:    jnp.ndarray        # (B, K, d, d)
    precs:   jnp.ndarray        # (B, K, d, d)
    log_norm_consts: jnp.ndarray# (B, K)

def _random_orthogonal(key, d):
    A = random.normal(key, (d, d))
    # QR → Q; ensure det>0
    q, r = jnp.linalg.qr(A)
    sign = jnp.sign(jnp.linalg.det(q))
    q = q * sign
    return q

def _random_spd(key, d):
    """Random SPD: R @ diag(eigs) @ R^T with eigs in [EIG_MIN, EIG_MAX]."""
    k1, k2 = random.split(key)
    R = _random_orthogonal(k1, d)
    eigs = EIG_MIN + (EIG_MAX - EIG_MIN) * random.uniform(k2, (d,))
    D = jnp.diag(eigs)
    C = R @ D @ R.T
    return C

def _make_weights(key, k):
    if k == 1:
        w = jnp.array([1.0, 0.0])
    else:
        # two components; clamp min weight
        r = random.uniform(key, (2,))
        r = jnp.clip(r, MIN_MIX_WEIGHT, 1.0)
        w = r / jnp.sum(r)
    return w

def sample_gmms(key, batch, d) -> GMM:
    """Build a batch of GMMs (K∈{1,2}, padded to 2) – no jit, plain Python loop."""
    k_choice_key, key = random.split(key)
    k_choices = random.choice(k_choice_key, K_CHOICES, shape=(batch,), replace=True)

    ws, means, covs, precs, log_norms = [], [], [], [], []
    for i in range(batch):
        key, k1, k2, k3, k4 = random.split(key, 5)
        K_i = int(k_choices[i])

        means_i = MEAN_RANGE * (2.0 * random.uniform(k1, (MAX_K, d)) - 1.0)
        C0, C1 = _random_spd(k2, d), _random_spd(k3, d)
        cov_i = jnp.stack([C0, C1], axis=0)                         # (2,d,d)
        w_i = _make_weights(k4, K_i)                                # (2,)

        prec_i = jnp.linalg.inv(cov_i)
        _, logdet = jnp.linalg.slogdet(cov_i)                       # (2,)
        log_norm_i = (d * 0.5) * jnp.log(2 * jnp.pi) + 0.5 * logdet # (2,)

        ws.append(w_i); means.append(means_i); covs.append(cov_i)
        precs.append(prec_i); log_norms.append(log_norm_i)

    return GMM(
        weights=jnp.stack(ws, axis=0),                 # (B,2)
        means=jnp.stack(means, axis=0),                # (B,2,d)
        covs=jnp.stack(covs, axis=0),                  # (B,2,d,d)
        precs=jnp.stack(precs, axis=0),                # (B,2,d,d)
        log_norm_consts=jnp.stack(log_norms, axis=0),  # (B,2)
    )

def sample_from_gmm(key, gmm: GMM, n: int):
    """Sample X: (B,N,d) from batched, padded-2 GMM."""
    B, K = gmm.weights.shape
    d    = gmm.means.shape[-1]
    k1, k2 = random.split(key)

    logits = jnp.log(jnp.clip(gmm.weights, 1e-12, 1.0))                 # (B,2)
    comp   = random.categorical(k1, logits[:, None, :], axis=-1, shape=(B, n))  # (B,N)

    L = jnp.linalg.cholesky(gmm.covs)                                   # (B,2,d,d)
    bidx = jnp.arange(B)[:, None]                                       # (B,1)

    means_sel = gmm.means[bidx, comp]                                   # (B,N,d)
    L_sel     = L[bidx, comp]                                           # (B,N,d,d)

    eps = random.normal(k2, (B, n, d))
    x = means_sel + jnp.einsum('bndd,bnd->bnd', L_sel, eps)             # (B,N,d)
    return x

def log_gauss_unnorm(dx, prec):
    # dx: (...,d), prec: (...,d,d) — returns -0.5 * (x-μ)^T Σ^{-1} (x-μ)
    qf = jnp.einsum('...i,...ij,...j->...', dx, prec, dx)
    return -0.5 * qf

def gmm_logpdf_components(x, gmm: GMM):
    """
    x: (B,N,d)
    return log [ w_k * N_k(x) ]: (B,N,K)
    """
    B, N, d = x.shape
    K = gmm.weights.shape[1]
    x_exp   = x[:, :, None, :]                      # (B,N,K,d)
    dx      = x_exp - gmm.means[:, None, :, :]      # (B,N,K,d)
    qf      = jnp.einsum('bnkd,bkij,bnkj->bnk', dx, gmm.precs, dx)  # (B,N,K)
    log_N   = -0.5 * qf - gmm.log_norm_consts[:, None, :]           # (B,N,K)
    return jnp.log(jnp.clip(gmm.weights[:, None, :], 1e-12, 1.0)) + log_N

def gmm_score(x, gmm: GMM):
    """
    ∇_x log f(x) = Σ_k γ_k(x) Σ_k^{-1}(μ_k - x), where γ_k ∝ w_k N_k(x)
    x: (B,N,d) → (B,N,d)
    """
    log_wNk = gmm_logpdf_components(x, gmm)               # (B,N,K)
    log_f   = jax.nn.logsumexp(log_wNk, axis=-1, keepdims=True)
    gamma   = jnp.exp(log_wNk - log_f)                    # (B,N,K)

    mu_minus_x = gmm.means[:, None, :, :] - x[:, :, None, :]         # (B,N,K,d)
    comp_score = jnp.einsum('bkij,bnkj->bnki', gmm.precs, mu_minus_x) # (B,N,K,d)
    weighted   = jnp.sum(gamma[..., None] * comp_score, axis=2)       # (B,N,d)
    return weighted

# -------------------------
# Transformer (permutation-equivariant, no pos enc)
# -------------------------
class TransformerBlock(nnx.Module):
    def __init__(self, d_model, nhead, ff_dim, *, seed=0, dtype=jnp.float32):
        self.ln1 = nnx.LayerNorm(
            num_features=d_model, feature_axes=(-1,), rngs=nnx.Rngs(seed+0), dtype=dtype
        )
        self.mha = nnx.MultiHeadAttention(
            num_heads=nhead,
            in_features=d_model,
            qkv_features=d_model,
            out_features=d_model,
            dropout_rate=0.0,
            decode=False,
            rngs=nnx.Rngs(seed+1),
            dtype=dtype,
        )
        self.ln2 = nnx.LayerNorm(
            num_features=d_model, feature_axes=(-1,), rngs=nnx.Rngs(seed+2), dtype=dtype
        )
        self.ff1 = nnx.Linear(d_model, ff_dim, rngs=nnx.Rngs(seed+3), dtype=dtype)
        self.ff2 = nnx.Linear(ff_dim, d_model, rngs=nnx.Rngs(seed+4), dtype=dtype)

    def __call__(self, x):                     # x: (B, N, d_model)
        h = x + self.mha(self.ln1(x))          # self-attn (q=k=v)
        z = jax.nn.gelu(self.ff1(self.ln2(h)))
        z = self.ff2(z)
        return h + z

class ScoreTransformer(nnx.Module):
    def __init__(self, d_in, d_model=128, nhead=8, num_layers=4, ff_dim=256, seed=0, dtype=jnp.float32):
        self.config = dict(d_in=d_in, d_model=d_model, nhead=nhead, num_layers=num_layers, ff_dim=ff_dim, seed=seed, dtype=str(dtype))
        self.inp = nnx.Linear(d_in, d_model, rngs=nnx.Rngs(seed+5), dtype=dtype)
        self.blocks = []
        for i in range(num_layers):
            self.blocks.append(
                TransformerBlock(d_model, nhead, ff_dim, seed=seed+10+i, dtype=dtype)
            )
        self.out = nnx.Linear(d_model, d_in, rngs=nnx.Rngs(seed+999), dtype=dtype)

    def __call__(self, x):                     # x: (B, N, d_in)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)                     # (B, N, d_in)

# -------------------------
# Training
# -------------------------
def make_batch(key, batch=BATCH, n=SEQ_LEN, d=DIM):
    k1, k2 = random.split(key)
    gmms = sample_gmms(k1, batch, d)
    X = sample_from_gmm(k2, gmms, n)             # (B,N,d)
    S = gmm_score(X, gmms)                       # (B,N,d)
    return X, S, gmms

def loss_fn(model, X, S):
    P = model(X)
    return jnp.mean((P - S) ** 2)

@nnx.jit  # handles nnx.Module state + optimizer
def train_step(model, optimizer, X, S):
    loss, grads = nnx.value_and_grad(loss_fn)(model, X, S)
    optimizer.update(grads)     # mutates model in-place
    return model, optimizer, loss

def train():
    model = ScoreTransformer(DIM, D_MODEL, NHEAD, LAYERS, FF_DIM, seed=SEED, dtype=jnp.float32)
    optimizer = nnx.Optimizer(model, optax.adamw(LR, weight_decay=WEIGHT_DECAY))
    key = random.PRNGKey(SEED)
    ema, beta = None, 0.98

    for step in tqdm(range(1, STEPS + 1)):
        key, kb = random.split(key)
        # make the batch OUTSIDE jit
        X, S, _ = make_batch(kb, BATCH, SEQ_LEN, DIM)   # (B,N,d)
        model, optimizer, loss_val = train_step(model, optimizer, X, S)
        # force sync so timings are real (optional)
        _ = jax.block_until_ready(loss_val)
        ema = loss_val if ema is None else beta * ema + (1 - beta) * loss_val
        if step % PRINT_EVERY == 0:
            print(f"step {step:5d}  loss {float(loss_val):.6f}  ema {float(ema):.6f}")
    return model

# -------------------------
# Eval / Visualization
# -------------------------
def sample_single_gmm(key, d=DIM, K: int = 2):
    # single-batch wrapper, returns (GMM with B=1), X (1,N,d)
    gmms = sample_gmms(key, batch=1, d=d)
    if K == 1:
        # force weights = [1,0] to ensure K=1
        gmms = GMM(
            weights=jnp.array([[1.0, 0.0]]),
            means=gmms.means,
            covs=gmms.covs,
            precs=gmms.precs,
            log_norm_consts=gmms.log_norm_consts,
        )
    X = sample_from_gmm(key, gmms, SEQ_LEN)
    return gmms, X

def eval_mse(model, key, K=2, n=SEQ_LEN, d=DIM):
    gmms, X = sample_single_gmm(key, d, K)
    S_true = gmm_score(X, gmms)
    S_pred = model(X)
    return float(jnp.mean((S_pred - S_true) ** 2))

@jax.jit
def predict(model, X):
    return model(X)

def visualize_scores(model, key, n=1000, m=20, K=None, d=DIM, fname=None):
    """
    Sample one GMM (K in {1,2}), draw n points, and plot -0.5 * (true score) and -0.5 * (pred score).
    Only does quiver if d==2.
    """
    if d != 2:
        print("Visualization is implemented for d=2.")
        return
    K = int(K) if K in (1, 2) else int(random.choice(key, K_CHOICES))
    k1, k2 = random.split(key)
    gmms = sample_gmms(k1, batch=1, d=d)
    if K == 1:
        gmms = GMM(
            weights=jnp.array([[1.0, 0.0]]),
            means=gmms.means,
            covs=gmms.covs,
            precs=gmms.precs,
            log_norm_consts=gmms.log_norm_consts,
        )
    # print params
    print("GMM parameters:")
    print(f"  K = {K}")
    print(f"  weights:\n{np.array(gmms.weights)[0]}")
    print(f"  means:\n{np.array(gmms.means)[0]}")
    print(f"  covariances (component 0):\n{np.array(gmms.covs)[0,0]}")
    print(f"  covariances (component 1):\n{np.array(gmms.covs)[0,1]}")

    X = sample_from_gmm(k2, gmms, n)             # (1,n,2)
    S_true = gmm_score(X, gmms)                  # (1,n,2)
    S_pred = model(X)                            # (1,n,2)

    x_np = np.array(X[0])
    t_np = np.array(-0.5 * S_true[0])
    p_np = np.array(-0.5 * S_pred[0])

    idx = np.random.permutation(n)[:m]
    pts, tru, prd = x_np[idx], t_np[idx], p_np[idx]

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
        label=f'pred: -½s, mse {np.mean((np.array(S_pred)-np.array(S_true))**2):.4f}',
        color='blue'
    )
    plt.axis('equal')
    plt.title('Universal Score Prediction with a Transformer (JAX/NNX)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=200)
    plt.show()

#%%
# -------------------------
# Run
# -------------------------
model = train()
print("Eval MSE on a fresh K=2 mixture:", eval_mse(model, random.PRNGKey(123), K=2, n=SEQ_LEN, d=DIM))

#%%
# Examples (for d=2):
visualize_scores(model, random.PRNGKey(1), n=1000, m=20, K=1, d=DIM)
visualize_scores(model, random.PRNGKey(2), n=1000, m=20, K=2, d=DIM)
visualize_scores(model, random.PRNGKey(3), n=1000, m=20, K=2, d=DIM)
visualize_scores(model, random.PRNGKey(4), n=1000, m=20, K=2, d=DIM)

#%%
"Save the checkpoint"

# checkpoints.py
import os, time, json
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from src import path as PATH  # uses PATH.MODELS

def _stamp():
    return time.strftime("%Y%m%d-%H%M%S")

def _ckpt_dir(d: int, tag: str | None = None) -> str:
    base = os.path.join(PATH.MODELS, "transformer")
    name = f"d{d}_{tag or _stamp()}"
    return os.path.join(base, name)

def save_transformer(model, d: int, tag: str | None = None, **kwargs) -> str:
    """
    Save NNX model state to `<MODELS>/transformer/d{d}_<tag or timestamp>/state`.
    Also writes meta.json with basic config for easy restore.
    kwargs are forwarded to orbax StandardCheckpointer.save (e.g., force=True).
    """
    ckpt_dir = _ckpt_dir(d, tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    # split graph/state
    _, state = nnx.split(model)

    # save state
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(os.path.join(ckpt_dir, "state"), state, **kwargs)

    # (optional) save minimal config to reinstantiate the module
    # Assumes your ScoreTransformer stored its config in `model.config`
    meta = getattr(model, "config", None)
    if meta is None:
        # fallback: best-effort probe
        meta = {
            "d_in": state["out"]["bias"].shape[0] if "out" in state else d,
            "dtype": str(getattr(model, "dtype", jnp.float32)),
        }
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return ckpt_dir

def load_transformer(ckpt_dir: str, construct_fn):
    """
    Restore a model from `<...>/state` using `construct_fn(**meta)` to build a fresh module.
    Returns the restored model.
    - `construct_fn` should create an uninitialized ScoreTransformer with the same config
      that you saved in meta.json (e.g., lambda **cfg: ScoreTransformer(**cfg)).
    """
    # read meta
    with open(os.path.join(ckpt_dir, "meta.json")) as f:
        meta = json.load(f)

    model = construct_fn(**meta)                      # fresh module with same shapes
    graphdef, abstract_state = nnx.split(model)

    checkpointer = ocp.StandardCheckpointer()
    state_restored = checkpointer.restore(os.path.join(ckpt_dir, "state"), abstract_state)

    restored = nnx.merge(graphdef, state_restored)
    return restored

# save
ckpt_dir = save_transformer(model, d=DIM, tag=f"runA_step{STEPS}", force=True)

# load later
# construct = lambda d_in, d_model, nhead, num_layers, ff_dim, seed, dtype, **_: \
#     ScoreTransformer(d_in, d_model, nhead, num_layers, ff_dim, seed=seed, dtype=getattr(jnp, dtype.split('.')[-1]))
# model = load_transformer(ckpt_dir, construct_fn=construct)

# %%
