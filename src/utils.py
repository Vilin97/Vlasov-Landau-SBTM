import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------------------
# Visualization utilities
# ------------------------------------------------------------------------------
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
        label=f"{label} score n={n:.0e} mse={float(jnp.mean((score - score_true)**2)):.3f}",
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

def plot_phase_space_snapshots(x_traj, v_traj, t_traj, L, title, outdir, fname, bins=[150,150]):
    num_snaps = len(x_traj)
    if num_snaps == 0:
        return None, None

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

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, fname)
    fig.savefig(path)

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
    v_new = v.at[:, 0].add(dt * E_at_particles)
    x_new = jnp.mod(x + dt * v[:, 0], box_length)
    E_new = update_electric_field(E, x, v, cells, eta, w, dt)
    return x_new, v_new, E_new


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
@jax.jit
def A_apply(dv, ds, gamma, eps=1e-14):
    v2 = jnp.sum(dv * dv, axis=-1, keepdims=True) + eps
    vg = v2 ** (gamma / 2)
    dvds = jnp.sum(dv * ds, axis=-1, keepdims=True)
    return vg * (v2 * ds - dvds * dv)


@partial(jax.jit, static_argnames=['num_cells'])
def collision(x, v, s, eta, gamma, num_cells, box_length, w):
    """
    Q_i = w Σ_{|x_i−x_j|≤η} ψ(x_i−x_j) A(v_i−v_j)(s_i−s_j)
    with linear-hat kernel ψ of width eta, periodic on [0,L].
    Complexity O(N η/L).
    """
    if x.ndim == 2:
        x = x[:, 0]
    N, d = v.shape
    M = num_cells

    cell = (jnp.floor(x / eta).astype(jnp.int32)) % M
    order = jnp.argsort(cell)
    x, v, s, cell = x[order], v[order], s[order], cell[order]

    counts = jnp.bincount(cell, length=M)
    starts = jnp.cumsum(jnp.concatenate([jnp.array([0]), counts[:-1]]))

    def centered_mod(y, L):
        return (y + L / 2) % L - L / 2

    def psi(y, eta, box_length):
        y = centered_mod(y, box_length)
        kernel = jnp.maximum(0.0, 1.0 - jnp.abs(y / eta))
        return kernel / eta

    def Q_single(i):
        xi, vi, si = x[i], v[i], s[i]
        ci = cell[i]
        acc = jnp.zeros(d)

        for c in ((ci - 1) % M, ci, (ci + 1) % M):
            start = starts[c]
            end = start + counts[c]

            def add_j(j, accj):
                ψ = psi(xi - x[j], eta, box_length)
                dv_ = vi - v[j]
                ds_ = si - s[j]
                return accj + ψ * A_apply(dv_, ds_, gamma)

            acc = lax.fori_loop(start, end, add_j, acc)
        return acc

    Q_sorted = jax.vmap(Q_single)(jnp.arange(N))
    rev = jnp.empty_like(order).at[order].set(jnp.arange(N))
    return w * Q_sorted[rev]
