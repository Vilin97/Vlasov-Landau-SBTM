import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os
import math

from src import loss
from flax import nnx
import optax

# ------------------------------------------------------------------------------
# Visualization utilities
# ------------------------------------------------------------------------------
def plot_U_quiver_pred(v, U, label, num_points=500, figsize=(5, 5)):
    assert v.shape == U.shape
    n = v.shape[0]
    step_sub = max(1, n // num_points)
    v_plot = v[::step_sub]
    U_plot = U[::step_sub]

    fig_quiver = plt.figure(figsize=figsize)
    plt.quiver(
        v_plot[:, 0],
        v_plot[:, 1],
        U_plot[:, 0],
        U_plot[:, 1],
        alpha=0.8,
        scale=5,
        angles="xy",
        scale_units="xy",
        label=label,
    )
    plt.scatter(v_plot[:, 0], v_plot[:, 1], s=2, alpha=0.3)
    plt.axis("equal")
    plt.xlabel("v1")
    plt.ylabel("v2")
    plt.title(f"Estimated flow U: {label}")
    plt.legend(loc="best")
    plt.tight_layout()
    return fig_quiver

def plot_score_quiver_pred(v, score, label, num_points=500, figsize=(5, 5)):
    assert v.shape == score.shape
    n = v.shape[0]
    step_sub = max(1, n // num_points)
    v_plot = v[::step_sub]
    score_plot = score[::step_sub]

    fig_quiver = plt.figure(figsize=figsize)
    plt.quiver(
        v_plot[:, 0],
        v_plot[:, 1],
        score_plot[:, 0],
        score_plot[:, 1],
        alpha=0.8,
        scale=5,
        angles="xy",
        scale_units="xy",
        label=label,
    )
    plt.scatter(v_plot[:, 0], v_plot[:, 1], s=2, alpha=0.3)
    plt.axis("equal")
    plt.xlabel("v1")
    plt.ylabel("v2")
    plt.title(f"Estimated scores: {label}")
    plt.legend(loc="best")
    plt.tight_layout()
    return fig_quiver

def plot_score_quiver(v, score, score_true, label, num_points=500, figsize=(5, 5)):
    assert v.shape == score.shape == score_true.shape
    n = v.shape[0]
    step_sub = max(1, n // num_points)
    v_plot = v[::step_sub]
    score_plot = score[::step_sub]
    score_true_plot = score_true[::step_sub]

    fig_quiver = plt.figure(figsize=figsize)
    plt.quiver(
        v_plot[:, 0],
        v_plot[:, 1],
        score_plot[:, 0],
        score_plot[:, 1],
        color="tab:blue",
        alpha=0.8,
        scale=5,
        angles="xy",
        scale_units="xy",
        label=f"{label} score n={n:.0e} mse={float(jnp.mean((score - score_true)**2)):.5f}",
    )
    plt.quiver(
        v_plot[:, 0],
        v_plot[:, 1],
        score_true_plot[:, 0],
        score_true_plot[:, 1],
        color="tab:red",
        alpha=0.5,
        scale=5,
        angles="xy",
        scale_units="xy",
        label="Reference score",
    )
    plt.scatter(v_plot[:, 0], v_plot[:, 1], s=2, c="k", alpha=0.3)
    plt.axis("equal")
    plt.xlabel("v1")
    plt.ylabel("v2")
    plt.title(f"Estimated scores: {label}")
    plt.legend(loc="best")
    plt.tight_layout()
    return fig_quiver

def visualize_initial(x, v, cells, E, rho, eta, L, spatial_density, v_target):
    """Visualize initial data and return the figure."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Histogram of x and desired density
    axs[0].hist(x, bins=50, density=True, alpha=0.4, label='Sampled $x$')
    x_grid = jnp.linspace(0, L, 200)
    axs[0].plot(x_grid, spatial_density(x_grid), 'r-', label='Target density')
    axs[0].plot(cells, rho / L, 'g-', label='$\\rho/L$')
    axs[0].set_title('Position $x$')
    axs[0].set_xlabel('$x$')
    axs[0].legend()

    # 2. Histogram of v and target mixture
    axs[1].hist(v, bins=50, density=True, alpha=0.4, label='Sampled $v$')
    v_grid = jnp.linspace(v.min() - 1, v.max() + 1, 200)
    axs[1].plot(v_grid, v_target(v_grid), 'r-', label='Target $f(v)$')
    axs[1].set_title('Velocity $v$')
    axs[1].set_xlabel('$v$')
    axs[1].legend()

    # 3. E, dE/dx, and rho
    axs[2].plot(cells, E, label='$E$')
    dE_dx = jnp.gradient(E, eta)
    axs[2].plot(cells, dE_dx, label='$dE/dx$')
    axs[2].plot(cells, rho - 1, label=r'$\rho - \rho_i$')
    axs[2].set_title('Field $E$, $dE/dx$, and $\\rho$')
    axs[2].set_xlabel('x')
    axs[2].legend()

    plt.tight_layout()
    return fig

def plot_phase_space_snapshots(x_traj, v_traj, t_traj, L, title, outdir=None, fname=None, bins=None, save=True):
    num_snaps = len(x_traj)
    if num_snaps == 0:
        return None, None
    if bins is None:
        k = int(math.sqrt(x_traj[0].shape[0] / 50))
        bins = [max(20, k), max(20, k)]

    cols = min(3, num_snaps)
    rows = int(np.ceil(num_snaps / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                            sharex=True, sharey=True)
    axs = np.array(axs).reshape(-1)

    v_all = np.concatenate([np.asarray(v_snap)[:, 0] for v_snap in v_traj])
    vmin, vmax = float(v_all.min()), float(v_all.max())

    last_img = None
    for i, (x_snap, v_snap, t_snap) in enumerate(zip(x_traj, v_traj, t_traj)):
        ax = axs[i]
        xs = np.asarray(x_snap) % float(L)
        vs = np.asarray(v_snap)[:, 0]
        _, _, _, img = ax.hist2d(
            xs, vs,
            bins=bins,
            range=[[0.0, float(L)], [vmin, vmax]],
            cmap="jet",
            density=True,
        )
        last_img = img
        ax.set_title(f"t = {t_snap:.1f}")
        ax.set_xlim(0.0, float(L))
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel("x")
        ax.set_ylabel("v")

    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    if last_img is not None:
        cbar = fig.colorbar(last_img, ax=axs.tolist(),
                            orientation="vertical", fraction=0.02, pad=0.02)
        cbar.set_label("Density")

    plt.suptitle(title)

    if save:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, fname)
        plt.savefig(path)
    else:
        path = None

    return fig, path

#------------------------------------------------------------------------------
# Rejection sampling
#------------------------------------------------------------------------------
def rejection_sample(key, density_fn, domain, max_value, num_samples=1):
    """Parallel rejection sampling on [domain[0], domain[1]]."""
    domain_width = domain[1] - domain[0]
    proposal_fn = lambda x: jnp.where((x >= domain[0]) & (x <= domain[1]), 1.0 / domain_width, 0.0)
    max_ratio = max_value / (1.0 / domain_width) * 1.2  # 20% margin
    key, key_propose, key_accept = jr.split(key, 3)

    num_candidates = int(num_samples * max_ratio * 2)
    candidates = jr.uniform(key_propose, minval=domain[0], maxval=domain[1], shape=(num_candidates,))
    proposal_values = proposal_fn(candidates)
    target_values = density_fn(candidates)

    accepted = jr.uniform(key_accept, (num_candidates,)) * max_ratio * proposal_values <= target_values
    samples = candidates[accepted]
    return samples[:num_samples]


#------------------------------------------------------------------------------
# PIC pieces (1D x, dv-D v)
#------------------------------------------------------------------------------
@jax.jit
def evaluate_charge_density(x, cells, eta, w):
    """ρ_j = w * Σ_p ψ_eta(X_p − cell_j) with ψ the hat kernel."""
    M = cells.size
    idx_f = x / eta - 0.5
    i0 = jnp.floor(idx_f).astype(jnp.int32) % M
    i1 = (i0 + 1) % M
    f = idx_f - jnp.floor(idx_f)
    w0, w1 = 1 - f, f
    counts = (
        jnp.zeros(M)
          .at[i0].add(w0)
          .at[i1].add(w1)
    )
    return w / eta * counts


@jax.jit
def evaluate_field_at_particles(E, x, cells, eta):
    """E(x) = Σ_j ψ(x − cell_j) E_j (linear-hat kernel, periodic)."""
    M = cells.size
    idx_f = x / eta - 0.5
    i0 = jnp.floor(idx_f).astype(jnp.int32) % M
    f = idx_f - jnp.floor(idx_f)
    i1 = (i0 + 1) % M
    return (1.0 - f) * E[i0] + f * E[i1]


@jax.jit
def update_electric_field(E, x, v, cells, eta, w, dt):
    """E_j^{n+1} = E_j^n - dt * w * Σ_i ψ(x_i - cell_j) v_i (linear-hat, periodic)."""
    M = cells.size
    idx_f = x / eta - 0.5
    i0 = jnp.floor(idx_f).astype(jnp.int32) % M
    i1 = (i0 + 1) % M
    f = idx_f - jnp.floor(idx_f)
    w0, w1 = 1 - f, f
    J = (
        jnp.zeros(M)
          .at[i0].add(w0 * v[:, 0])
          .at[i1].add(w1 * v[:, 0])
    )
    dEdt = w / eta * J
    return (E - dt * dEdt).astype(E.dtype)


@jax.jit
def vlasov_step(x, v, E, cells, eta, dt, box_length, w):
    """Forward Euler time stepping for Vlasov part."""
    E_at_particles = evaluate_field_at_particles(E, x, cells, eta)
    v = v.at[:, 0].add(dt * E_at_particles)
    x = jnp.mod(x + dt * v[:, 0], box_length)
    E = update_electric_field(E, x, v, cells, eta, w, dt)
    return x, v, E


#------------------------------------------------------------------------------
# KDE in phase space and Landau collision
#------------------------------------------------------------------------------
"KDE score model for 1D space"
def _silverman_bandwidth(v, eps=1e-12):
        n, dv = v.shape
        sigma = jnp.std(v, axis=0, ddof=1) + eps
        return sigma * n ** (-1.0 / (dv + 4.0))  # (dv,)

@partial(jax.jit, static_argnames=['max_ppc'])
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
        Xi = Xc[c]                    # (max_ppc,)
        Vi = Vc[c]                    # (max_ppc,dv)
        Ui = Uc[c]
        Ui2 = U2c[c]                  # (max_ppc,1)
        mask_i = maskc[c][:, None]    # (max_ppc,1)

        c0 = (c - 1) % M
        c1 = c
        c2 = (c + 1) % M
        Xj = jnp.concatenate([Xc[c0], Xc[c1], Xc[c2]], axis=0)        # (3*max_ppc,)
        Vj = jnp.concatenate([Vc[c0], Vc[c1], Vc[c2]], axis=0)        # (3*max_ppc,dv)
        Uj = jnp.concatenate([Uc[c0], Uc[c1], Uc[c2]], axis=0)
        Uj2 = jnp.concatenate([U2c[c0], U2c[c1], U2c[c2]], axis=0)    # (3*max_ppc,1)
        mask_j = jnp.concatenate(
            [maskc[c0], maskc[c1], maskc[c2]], axis=0
        )[:, None]                                                     # (3*max_ppc,1)

        dx = Xi[:, None] - Xj[None, :]
        dx = (dx + 0.5 * L) % L - 0.5 * L
        psi = jnp.maximum(0.0, 1.0 - jnp.abs(dx) / eta)               # hat in x

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

# this is ~11 times faster than score_kde_blocked with n=1e5 and M=50
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
    max_ppc = ((m + 99) // 100) * 100  # next multiple of 100 >= m

    return _score_kde_local_impl(x, v, cells, eta, eps, hv, max_ppc)


def scaled_score_kde(x, v, cells, eta, eta_scale=4, hv_scale=4, output_scale=1.3, **kwargs):
    """Empirically tuned scaled KDE score."""
    hv = _silverman_bandwidth(v) * hv_scale
    s_kde = score_kde(x, v, cells, eta * eta_scale, hv=hv, **kwargs) * output_scale
    return s_kde

#------------------------------------------------------------------------------
# Landau collision operator
#------------------------------------------------------------------------------
def compute_window_params(N, eta, box_length, bucket_size=100, safety_factor=1.2):
    """
    Calculates the required window size based on density and buckets it.
    
    Args:
        N: Number of particles
        eta: Interaction radius
        box_length: Domain size
        bucket_size: Round up to multiples of this (prevents frequent recompilation)
        safety_factor: Multiplier to account for particle clustering (non-uniformity)
    """
    # Average density of particles
    density = N / box_length
    
    # Expected neighbors in range [-eta, +eta] (so 2 * eta)
    # We only need 'window_size' which is the radius (neighbors on ONE side)
    avg_neighbors_radius = density * eta
    
    # Add safety margin for clustering
    target_size = avg_neighbors_radius * safety_factor
    
    # Round up to nearest bucket_size (e.g., 100)
    # Formula: ceil(x / 100) * 100
    window_size = math.ceil(target_size / bucket_size) * bucket_size
    
    # Clamp: Window cannot be larger than N//2 (wrapping around the world)
    window_size = min(int(window_size), N // 2)
    
    # Ensure at least 1 to avoid shape errors
    return max(window_size, 1)

@jax.jit
def A_apply(dv, ds, gamma, eps=1e-14):
    v2 = jnp.sum(dv * dv, axis=-1, keepdims=True) + eps
    vg = v2 ** (gamma / 2)
    dvds = jnp.sum(dv * ds, axis=-1, keepdims=True)
    return vg * (v2 * ds - dvds * dv)

@partial(jax.jit, static_argnames=['window_size'])
def collision_rolling(x, v, s, eta, gamma, box_length, w, window_size):
    """
    O(N) Memory implementation using Vectorized Rolling.
    """
    if x.ndim == 2:
        x = x[:, 0]
    
    N, d = v.shape

    # 1. Sort (Essential for windowing)
    order = jnp.argsort(x)
    x_sorted = x[order]
    v_sorted = v[order]
    s_sorted = s[order]

    # 2. Define the loop body
    # Instead of a massive matrix, we compute one "offset layer" at a time.
    def body_fn(i, val):
        Q_accum = val
        
        # Map loop index i (0 to 2*window) to offset (-window to +window)
        offset = i - window_size
        
        # Fetch Neighbor Data using Roll
        # jnp.roll is O(N) and very fast on GPU
        x_neighbor = jnp.roll(x_sorted, shift=offset, axis=0)
        v_neighbor = jnp.roll(v_sorted, shift=offset, axis=0)
        s_neighbor = jnp.roll(s_sorted, shift=offset, axis=0)

        # Periodic Distance
        dx = x_sorted - x_neighbor
        dx = (dx + box_length / 2) % box_length - box_length / 2
        dist = jnp.abs(dx)
        
        # Physics & Masking
        # We multiply by (offset != 0) to avoid self-interaction
        mask = (dist <= eta) & (dist > 0.0) & (offset != 0)
        
        psi = jnp.maximum(0.0, 1.0 - dist / eta) / eta
        psi = psi * mask

        # Since we are 1D in the loop, we need to expand dims for broadcasting
        # psi is (N,), we need (N, 1) to multiply against (N, d)
        psi = psi[:, None]

        dv = v_sorted - v_neighbor
        ds = s_sorted - s_neighbor
        
        interaction = A_apply(dv, ds, gamma)
        
        # Accumulate result
        return Q_accum + interaction * psi

    # 3. Run the Loop
    # We iterate 2*window_size + 1 times.
    # Unlike your original code, the work inside here is perfectly balanced (size N).
    Q_init = jnp.zeros_like(v)
    Q_sorted = lax.fori_loop(0, 2 * window_size + 1, body_fn, Q_init)

    # 4. Unsort
    rev = jnp.empty_like(order).at[order].set(jnp.arange(N))
    return w * Q_sorted[rev]

# on rtx6k, with fp64, n=1e6, M=100, dv=2, takes ~16s and 448Mb memory
def collision(x, v, s, eta, gamma, box_length, w):
    """
    Driver function that calculates window size and dispatches to JIT kernel.
    """
    n = v.shape[0]
    
    # 1. Pre-compute the window size (Python int)
    # This is fast and happens outside JAX.
    win_size = compute_window_params(n, eta, box_length)
    
    # 2. Call the JIT-compiled kernel
    return collision_rolling(x, v, s, eta, gamma, box_length, w, window_size=win_size)

#------------------------------------------------------------------------------
# SBTM helpers
#------------------------------------------------------------------------------
def mse_loss(model, batch):
    x, v, s = batch
    pred = model(x, v)
    return loss.mse(pred, s)

def ism_loss(model, batch, key):
    x, v = batch
    return loss.implicit_score_matching_loss(model, x, v, key=key)

@nnx.jit
def supervised_step(model, optimizer, batch):
    loss_val, grads = nnx.value_and_grad(mse_loss)(model, batch)
    optimizer.update(grads)
    return loss_val

@nnx.jit
def score_step(model, optimizer, batch, key):
    loss_val, grads = nnx.value_and_grad(ism_loss)(model, batch, key)
    optimizer.update(grads)
    return loss_val

def train_initial_model(model, x, v, score, batch_size, num_epochs, abs_tol, lr, verbose=False):
    optimizer = nnx.Optimizer(model, optax.adamw(lr))
    n = x.shape[0]
    for epoch in range(num_epochs):
        full_loss = mse_loss(model, (x, v, score))
        if verbose:
            print(f"Epoch {epoch}: loss = {full_loss:.5f}")
        if full_loss < abs_tol:
            if verbose:
                print(f"Stopping at epoch {epoch} with loss {full_loss:.5f} < {abs_tol}")
            break
        key = jr.PRNGKey(epoch)
        perm = jr.permutation(key, n)
        x_sh, v_sh, s_sh = x[perm], v[perm], score[perm]
        for i in range(0, n, batch_size):
            batch = (
                x_sh[i:i+batch_size],
                v_sh[i:i+batch_size],
                s_sh[i:i+batch_size],
            )
            supervised_step(model, optimizer, batch)

def train_score_model(model, optimizer, x, v, key, batch_size, num_batch_steps):
    n = x.shape[0]
    losses = []
    batch_count = 0
    while batch_count < num_batch_steps:
        key, subkey = jr.split(key)
        perm = jr.permutation(subkey, n)
        x = x[perm]
        v = v[perm]
        start = 0
        while start < n and batch_count < num_batch_steps:
            end = min(start + batch_size, n)
            batch = (x[start:end], v[start:end])
            key, subkey = jr.split(key)
            loss_val = score_step(model, optimizer, batch, subkey)
            losses.append(loss_val)
            start = end
            batch_count += 1
    return losses