#%%

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from functools import partial
import matplotlib as mpl

PPI = 300  # pixels per inch
mpl.rcParams["figure.dpi"] = PPI
mpl.rcParams["savefig.dpi"] = PPI

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_true_score_quiver(v, score_true, label, num_points=500, figsize=(5, 5), scale=20):
    n = v.shape[0]
    step_sub = max(1, n // num_points)
    v_plot = v[::step_sub]
    score_true_plot = score_true[::step_sub]

    fig_quiver = plt.figure(figsize=figsize)
    plt.quiver(
        np.asarray(v_plot[:, 0]),
        np.asarray(v_plot[:, 1]),
        np.asarray(score_true_plot[:, 0]),
        np.asarray(score_true_plot[:, 1]),
        color="tab:red",
        alpha=0.8,
        scale=scale,
        angles="xy",
        scale_units="xy",
        label=f"{label} (n={n})",
    )
    plt.scatter(
        np.asarray(v_plot[:, 0]),
        np.asarray(v_plot[:, 1]),
        s=2,
        alpha=0.3,
    )
    plt.axis("equal")
    plt.xlabel("v1")
    plt.ylabel("v2")
    plt.title(f"True scores: {label}")
    plt.legend(loc="best")
    plt.tight_layout()
    return fig_quiver


def plot_score_quiver(v, score, score_true, label, num_points=500, figsize=(5, 5), scale=20):
    assert v.shape == score.shape == score_true.shape
    n = v.shape[0]
    step_sub = max(1, n // num_points)
    v_plot = v[::step_sub]
    score_plot = score[::step_sub]
    score_true_plot = score_true[::step_sub]

    mse = float(jnp.mean((score - score_true) ** 2))

    fig_quiver = plt.figure(figsize=figsize)
    plt.quiver(
        np.asarray(v_plot[:, 0]),
        np.asarray(v_plot[:, 1]),
        np.asarray(score_plot[:, 0]),
        np.asarray(score_plot[:, 1]),
        color="tab:blue",
        alpha=0.8,
        scale=scale,
        angles="xy",
        scale_units="xy",
        label=f"{label} score n={n:.0e} mse={mse:.5f}",
    )
    plt.quiver(
        np.asarray(v_plot[:, 0]),
        np.asarray(v_plot[:, 1]),
        np.asarray(score_true_plot[:, 0]),
        np.asarray(score_true_plot[:, 1]),
        color="tab:red",
        alpha=0.5,
        scale=scale,
        angles="xy",
        scale_units="xy",
        label="Reference score",
    )
    plt.scatter(
        np.asarray(v_plot[:, 0]),
        np.asarray(v_plot[:, 1]),
        s=2,
        alpha=0.3,
    )
    plt.axis("equal")
    plt.xlabel("v1")
    plt.ylabel("v2")
    plt.title(f"Estimated scores: {label}")
    plt.legend(loc="best")
    plt.tight_layout()
    return fig_quiver


# ---------------------------------------------------------------------
# KDE score model in (x, v) with 1D x and dv-dim v
# ---------------------------------------------------------------------
def _silverman_bandwidth(v, eps=1e-12):
    n, dv = v.shape
    sigma = jnp.std(v, axis=0, ddof=1) + eps
    return sigma * n ** (-1.0 / (dv + 4.0))


@partial(jax.jit, static_argnames=["max_ppc"])
def _score_kde_local_impl(x, v, cells, eta, eps=1e-12, hv=None, max_ppc=4096):
    if x.ndim == 2:
        x = x[:, 0]
    if hv is None:
        hv = _silverman_bandwidth(v, eps)

    n, dv = v.shape
    M = cells.size
    L = eta * M
    inv_hv = 1.0 / hv
    inv_hv2 = inv_hv ** 2

    idx = jnp.floor(x / eta).astype(jnp.int32) % M
    order = jnp.argsort(idx)
    x_s = x[order]
    v_s = v[order]
    idx_s = idx[order]

    counts = jnp.bincount(idx_s, length=M).astype(jnp.int32)
    cell_ofs = jnp.cumsum(
        jnp.concatenate([jnp.array([0], dtype=jnp.int32), counts[:-1]]),
        dtype=jnp.int32,
    )

    Xc = jnp.zeros((M, max_ppc), x.dtype)
    Vc = jnp.zeros((M, max_ppc, dv), v.dtype)
    maskc = jnp.zeros((M, max_ppc), x.dtype)
    idx_map = -jnp.ones((M, max_ppc), jnp.int32)
    ar_ppc = jnp.arange(max_ppc, dtype=jnp.int32)

    def fill_cell(c, carry):
        Xc, Vc, maskc, idx_map = carry
        cnt = counts[c]
        base = cell_ofs[c]
        valid = ar_ppc < cnt
        gidx = base + ar_ppc
        gidx = jnp.where(valid, gidx, 0)
        Xc = Xc.at[c].set(jnp.where(valid, x_s[gidx], 0.0))
        Vc = Vc.at[c].set(jnp.where(valid[:, None], v_s[gidx], 0.0))
        maskc = maskc.at[c].set(valid.astype(x.dtype))
        idx_map = idx_map.at[c].set(jnp.where(valid, gidx, -1))
        return Xc, Vc, maskc, idx_map

    Xc, Vc, maskc, idx_map = lax.fori_loop(
        0, M, fill_cell, (Xc, Vc, maskc, idx_map)
    )

    Uc = Vc * inv_hv
    U2c = jnp.sum(Uc * Uc, axis=-1, keepdims=True)

    Zc = jnp.zeros((M, max_ppc, 1), v.dtype)
    Mc = jnp.zeros((M, max_ppc, dv), v.dtype)

    def body_cell(c, carry):
        Zc, Mc = carry
        Xi = Xc[c]
        Vi = Vc[c]
        Ui = Uc[c]
        Ui2 = U2c[c]
        mask_i = maskc[c][:, None]

        c0 = (c - 1) % M
        c1 = c
        c2 = (c + 1) % M
        Xj = jnp.concatenate([Xc[c0], Xc[c1], Xc[c2]], axis=0)
        Vj = jnp.concatenate([Vc[c0], Vc[c1], Vc[c2]], axis=0)
        Uj = jnp.concatenate([Uc[c0], Uc[c1], Uc[c2]], axis=0)
        Uj2 = jnp.concatenate([U2c[c0], U2c[c1], U2c[c2]], axis=0)
        mask_j = jnp.concatenate([maskc[c0], maskc[c1], maskc[c2]], axis=0)[
            :, None
        ]

        dx = Xi[:, None] - Xj[None, :]
        dx = (dx + 0.5 * L) % L - 0.5 * L
        psi = jnp.maximum(0.0, 1.0 - jnp.abs(dx) / eta)

        G = Ui @ Uj.T
        K = jnp.exp(G - 0.5 * Ui2 - 0.5 * Uj2.T)

        mask = mask_i * mask_j.T
        w = (psi * K + eps) * mask

        Z_local = jnp.sum(w, axis=1, keepdims=True) * mask_i
        M_local = (w @ Vj) * mask_i

        Zc = Zc.at[c].set(Z_local)
        Mc = Mc.at[c].set(M_local)
        return Zc, Mc

    Zc, Mc = lax.fori_loop(0, M, body_cell, (Zc, Mc))

    idx_flat = idx_map.reshape(-1)
    Z_flat = Zc.reshape(-1, 1)
    M_flat = Mc.reshape(-1, dv)
    valid = idx_flat >= 0
    idx_valid = jnp.where(valid, idx_flat, 0)
    Z_contrib = Z_flat * valid[:, None]
    M_contrib = M_flat * valid[:, None]

    Zs = jnp.zeros((n, 1), v.dtype)
    Ms = jnp.zeros((n, dv), v.dtype)
    Zs = Zs.at[idx_valid].add(Z_contrib)
    Ms = Ms.at[idx_valid].add(M_contrib)

    inv_order = jnp.empty_like(order)
    inv_order = inv_order.at[order].set(jnp.arange(n, dtype=order.dtype))

    Z = Zs[inv_order]
    M = Ms[inv_order]

    Z_safe = jnp.where(Z > 0, Z, eps)
    mu = M / Z_safe
    return (mu - v) * inv_hv2


def score_kde(x, v, cells, eta, eps=1e-12, hv=None):
    if hv is None:
        hv = _silverman_bandwidth(v, eps)
    if x.ndim == 2:
        x1d = x[:, 0]
    else:
        x1d = x
    M = cells.size
    idx = jnp.floor(x1d / eta).astype(jnp.int32) % M
    counts = jnp.bincount(idx, length=M)
    max_count = int(jax.device_get(jnp.max(counts)))
    m = max(1, max_count)
    max_ppc = ((m + 99) // 100) * 100
    return _score_kde_local_impl(x, v, cells, eta, eps, hv, max_ppc)


# ---------------------------------------------------------------------
# True density f(x, v1, v2) = g(x, v1) * h(v2)
# ---------------------------------------------------------------------
C_NOISE = 0.08


def grad_log_g(z):
    x = z[:, 0]
    v1 = z[:, 1]
    r = jnp.sqrt(x**2 + v1**2) + 1e-8
    coef = -2.0 * (r - 1.0) / (C_NOISE * r)
    gx = coef * x
    gv1 = coef * v1
    return jnp.stack([gx, gv1], axis=1)


def true_score_v(x, v):
    v1 = v[:, 0]
    v2 = v[:, 1]
    r = jnp.sqrt(x**2 + v1**2) + 1e-8
    coef = -2.0 * (r - 1.0) / (C_NOISE * r)
    s1 = coef * v1
    s2 = -v2
    return jnp.stack([s1, s2], axis=1)


def sample_noisy_circle(key, n, num_steps=800, step=1e-2):
    key_init, key_scan = jr.split(key)
    z0 = jr.normal(key_init, (n, 2))

    def ula_step(z, key):
        noise = jr.normal(key, z.shape)
        z = z + 0.5 * step * grad_log_g(z) + jnp.sqrt(step) * noise
        return z, None

    keys = jr.split(key_scan, num_steps)
    z, _ = lax.scan(ula_step, z0, keys)
    return z


def sample_from_f(key, n):
    key_z, key_v2 = jr.split(key)
    z = sample_noisy_circle(key_z, n)
    x = z[:, 0]
    v1 = z[:, 1]
    v2 = jr.normal(key_v2, (n,))
    v = jnp.stack([v1, v2], axis=1)
    return x, v


# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------
def run_experiment(n=10_000, x_bucket_size=0.5, scale=20):
    key = jr.PRNGKey(0)
    x, v = sample_from_f(key, n)
    score_true = true_score_v(x, v)

    # scatter (x, v1) and (v1, v2)
    x_np = np.asarray(x)
    v_np = np.asarray(v)

    plt.figure(figsize=(5, 5))
    plt.scatter(x_np, v_np[:, 0], s=2, alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("v1")
    plt.title("(x, v1) samples")
    plt.tight_layout()

    plt.figure(figsize=(5, 5))
    plt.scatter(v_np[:, 0], v_np[:, 1], s=2, alpha=0.5)
    plt.xlabel("v1")
    plt.ylabel("v2")
    plt.title("(v1, v2) samples")
    plt.axis("equal")
    plt.tight_layout()

    # KDE setup
    L = 8.0
    eta = 0.1
    M = int(L / eta)
    cells = jnp.arange(M) * eta
    x_kde = (x + L / 2.0) % L

    # global KDE score
    score_kde_global = score_kde(x_kde, v, cells, eta)

    # global true-score quiver
    plot_true_score_quiver(v, score_true, label="global", num_points=500)

    # global KDE vs true
    plot_score_quiver(v, score_kde_global, score_true, label="global KDE", num_points=500)

    # bucketed plots in one figure
    x_min, x_max = -3.0, 3.0
    edges = np.arange(x_min, x_max, x_bucket_size)

    buckets = []
    for left in edges:
        right = left + x_bucket_size
        mask = (x_np >= left) & (x_np < right)
        if mask.sum() >= 50:
            buckets.append((left, right, mask))

    B = len(buckets)
    if B == 0:
        return

    ncols = min(2, B)
    nrows = int(np.ceil(B / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8 * nrows), squeeze=False)

    for k, (left, right, mask) in enumerate(buckets):
        i, j = divmod(k, ncols)
        ax = axes[i, j]

        x_b = jnp.asarray(x_np[mask])
        v_b = jnp.asarray(v_np[mask])
        score_true_b = true_score_v(x_b, v_b)

        x_b_kde = (x_b + L / 2.0) % L
        score_kde_b = score_kde(x_b_kde, v_b, cells, eta)

        n_b = v_b.shape[0]
        step_sub = max(1, n_b // 500)
        v_plot = v_b[::step_sub]
        score_plot = score_kde_b[::step_sub]
        score_true_plot = score_true_b[::step_sub]

        mse_b = float(jnp.mean((score_kde_b - score_true_b) ** 2))

        ax.quiver(
            np.asarray(v_plot[:, 0]),
            np.asarray(v_plot[:, 1]),
            np.asarray(score_plot[:, 0]),
            np.asarray(score_plot[:, 1]),
            color="tab:blue",
            alpha=0.8,
            scale=scale,
            angles="xy",
            scale_units="xy",
        )
        ax.quiver(
            np.asarray(v_plot[:, 0]),
            np.asarray(v_plot[:, 1]),
            np.asarray(score_true_plot[:, 0]),
            np.asarray(score_true_plot[:, 1]),
            color="tab:red",
            alpha=1,
            scale=scale,
            angles="xy",
            scale_units="xy",
        )
        ax.scatter(
            np.asarray(v_plot[:, 0]),
            np.asarray(v_plot[:, 1]),
            s=2,
            alpha=0.3,
        )
        ax.set_title(f"[{left:.1f}, {right:.1f}) mse={mse_b:.4f}")
        ax.set_xlabel("v1")
        ax.set_ylabel("v2")
        ax.set_aspect("equal", "box")

    # hide unused axes
    for k in range(B, nrows * ncols):
        i, j = divmod(k, ncols)
        fig.delaxes(axes[i, j])

    fig.suptitle("Bucketed KDE vs true scores in (v1, v2)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


#%% run
run_experiment()
plt.show()
