# Two-stream instability with CLI + wandb logging

import argparse
import os
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import numpy as np

#------------------------------------------------------------------------------
# Utilities
#------------------------------------------------------------------------------
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
def _silverman_bandwidth(v, eps=1e-12):
    n, dv = v.shape
    sigma = jnp.std(v, axis=0, ddof=1) + eps
    return sigma * n ** (-1.0 / (dv + 4.0))  # (dv,)


@partial(jax.jit, static_argnames=['ichunk', 'jchunk'])
def score_kde(x, v, cells, eta, eps=1e-12, hv=None, ichunk=2048, jchunk=2048):
    if x.ndim == 2:
        x = x[:, 0]
    if hv is None:
        hv = _silverman_bandwidth(v, eps)
    L = eta * cells.size
    n, dv = v.shape
    inv_hv = 1.0 / hv
    inv_hv2 = inv_hv ** 2

    ni = (n + ichunk - 1) // ichunk
    nj = (n + jchunk - 1) // jchunk
    n_pad = ni * ichunk
    pad = n_pad - n

    x_pad = jnp.pad(x, (0, pad))
    v_pad = jnp.pad(v, ((0, pad), (0, 0)))
    u_pad = v_pad * inv_hv

    Zp = jnp.zeros((n_pad, 1), v.dtype)
    Mp = jnp.zeros((n_pad, dv), v.dtype)

    ar_i = jnp.arange(ichunk)
    ar_j = jnp.arange(jchunk)

    def loop_j(tj, carry2):
        Zi_, Mi_, Ri, Ui, Vi, Ui2 = carry2
        j0 = tj * jchunk
        mj = jnp.minimum(jchunk, n - j0)

        Rj = lax.dynamic_slice(x_pad, (j0,), (jchunk,))
        Uj = lax.dynamic_slice(u_pad, (j0, 0), (jchunk, dv))
        Vj = lax.dynamic_slice(v_pad, (j0, 0), (jchunk, dv))
        Uj2 = jnp.sum(Uj * Uj, axis=1, keepdims=True).T
        mask_j = (ar_j < mj).astype(v.dtype).reshape(1, jchunk)

        dx = Ri[:, None] - Rj[None, :]
        dx = (dx + 0.5 * L) % L - 0.5 * L
        psi = jnp.clip(1.0 - jnp.abs(dx) / eta, 0.0, 1.0)

        G = Ui @ Uj.T
        Kj = jnp.exp(G - 0.5 * Ui2 - 0.5 * Uj2)

        w = (psi * Kj + eps) * mask_j
        Zi_ = Zi_ + jnp.sum(w, axis=1, keepdims=True)
        Mi_ = Mi_ + w @ Vj
        return Zi_, Mi_, Ri, Ui, Vi, Ui2

    def loop_i(ti, carry):
        Zc, Mc = carry
        i0 = ti * ichunk
        mi = jnp.minimum(ichunk, n - i0)

        Ri = lax.dynamic_slice(x_pad, (i0,), (ichunk,))
        Ui = lax.dynamic_slice(u_pad, (i0, 0), (ichunk, dv))
        Vi = lax.dynamic_slice(v_pad, (i0, 0), (ichunk, dv))
        Ui2 = jnp.sum(Ui * Ui, axis=1, keepdims=True)

        Zi = jnp.zeros((ichunk, 1), v.dtype)
        Mi = jnp.zeros((ichunk, dv), v.dtype)

        Zi, Mi, *_ = lax.fori_loop(0, nj, loop_j, (Zi, Mi, Ri, Ui, Vi, Ui2))

        mask_i = (ar_i < mi).astype(v.dtype).reshape(ichunk, 1)
        Zi = Zi * mask_i
        Mi = Mi * mask_i

        Zc = lax.dynamic_update_slice(Zc, Zi, (i0, 0))
        Mc = lax.dynamic_update_slice(Mc, Mi, (i0, 0))
        return Zc, Mc

    Zp, Mp = lax.fori_loop(0, ni, loop_i, (Zp, Mp))
    Z = Zp[:n]
    M = Mp[:n]
    mu = M / Z
    return (mu - v) * inv_hv2


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


#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="1D-dvV two-stream Vlasov–Landau PIC solver")
    p.add_argument("--n", type=int, default=10**6, help="Number of particles (must be even)")
    p.add_argument("--M", type=int, default=100, help="Number of spatial cells")
    p.add_argument("--dt", type=float, default=0.05, help="Time step")
    p.add_argument("--gpu", type=int, default=0, help="CUDA_VISIBLE_DEVICES index")
    p.add_argument("--fp32", action="store_true", help="Use float32 instead of float64")
    p.add_argument("--dv", type=int, default=2, help="Velocity dimension")
    p.add_argument("--final_time", type=float, default=50.0, help="Final simulation time")
    p.add_argument("--C", type=float, default=0.08, help="Collision strength (C=0 => collisionless)")
    p.add_argument("--alpha", type=float, default=1/200, help="Amplitude of initial density perturbation")
    p.add_argument("--k", type=float, default=1/5, help="Wave number k")
    p.add_argument("--c", type=float, default=2.4, help="Beam speed for two-stream")
    p.add_argument("--score_method", type=str, default="scaled_kde", choices=["kde", "scaled_kde"])

    p.add_argument("--wandb_project", type=str, default="vlasov_two_stream", help="wandb project name")
    p.add_argument("--wandb_run_name", type=str, default="two_stream", help="wandb run name")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--log_every", type=int, default=1, help="Log every k steps")
    return p.parse_args()


def main():
    args = parse_args()

    assert args.n % 2 == 0, "--n must be even for two-stream initialization"

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    jax.config.update("jax_enable_x64", not args.fp32)

    wandb_run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )

    # Parameters
    seed = 42
    q = 1.0
    dv = args.dv
    alpha = args.alpha
    k = args.k
    c = args.c
    L = 2 * jnp.pi / k
    n = args.n
    M = args.M
    dt = args.dt
    eta = L / M
    cells = (jnp.arange(M) + 0.5) * eta
    w = q * L / n
    C = args.C
    gamma = -dv

    key_v, key_x = jr.split(jr.PRNGKey(seed), 2)
    n_half = n // 2

    # Two-stream initial velocities: v1 ~ N(0, I), shifted by +c; v2 = -v1
    v1 = jr.normal(key_v, (n_half, dv))
    v1 -= jnp.mean(v1, axis=0)
    v1 = v1.at[:, 0].add(c)
    v2 = -v1
    v = jnp.vstack([v1, v2])

    score_method = args.score_method
    if score_method == "kde":
        score_fn = score_kde
    elif score_method == "scaled_kde":
        score_fn = scaled_score_kde
    else:
        raise ValueError(f"Unknown score method: {score_method}")

    print(f"Args: {args}")

    def spatial_density(x):
        return (1 + alpha * jnp.cos(k * x)) / (2 * jnp.pi / k)

    # Ideal target in v: symmetric two-stream mixture
    def v_target(vv):
        return 0.5 * (
            jax.scipy.stats.norm.pdf(vv, -c, 1.0) +
            jax.scipy.stats.norm.pdf(vv,  c, 1.0)
        )

    max_value = jnp.max(spatial_density(cells))
    domain = (0.0, float(L))
    x = rejection_sample(key_x, spatial_density, domain, max_value=max_value, num_samples=n)

    rho = evaluate_charge_density(x, cells, eta, w)
    E = jnp.cumsum(rho - 1) * eta
    E = E - jnp.mean(E)

    fig_init = visualize_initial(x, v[:, 0], cells, E, rho, eta, L, spatial_density, v_target)
    wandb.log({"initial_state": wandb.Image(fig_init)}, step=0)
    plt.show()
    plt.close(fig_init)

    final_time = args.final_time
    num_steps = int(final_time / dt)
    t = 0.0
    E_L2 = [jnp.sqrt(jnp.sum(E ** 2) * eta)]

    print(
        f"Two-stream with n={n:.0e}, M={M}, dt={dt}, eta={float(eta):.4f}, "
        f"score_method={args.score_method}, dv={dv}, C={C}, alpha={alpha}, k={k}, c={c}, gpu={args.gpu}, fp32={args.fp32}"
    )

    # Quiver of scores before time stepping
    s_kde = score_fn(x, v, cells, eta)
    def two_stream_score_v(v, c):
        # v: (n, dv)
        v1 = v[:, 0]                      # (n,)
        r = jnp.tanh(c * v1)              # (n,)
        mu = jnp.zeros(v.shape[1], v.dtype).at[0].set(c)  # (dv,)
        return -v + r[:, None] * mu       # (n, dv)
    s_true = two_stream_score_v(v, c)


    step_sub = max(1, n // 500)
    v_plot = v[::step_sub]
    s_kde_plot = s_kde[::step_sub]
    s_true_plot = s_true[::step_sub]

    fig_quiver = plt.figure(figsize=(6, 6))
    plt.quiver(
        v_plot[:, 0],
        v_plot[:, 1],
        s_kde_plot[:, 0],
        s_kde_plot[:, 1],
        color="tab:blue",
        alpha=0.8,
        scale=5,
        angles="xy",
        scale_units="xy",
        label=f"KDE score n={n:.1e} mse={float(jnp.mean((s_kde - s_true)**2)):.3f}",
    )
    plt.quiver(
        v_plot[:, 0],
        v_plot[:, 1],
        s_true_plot[:, 0],
        s_true_plot[:, 1],
        color="tab:red",
        alpha=0.5,
        scale=5,
        angles="xy",
        scale_units="xy",
        label="Reference score (-v)",
    )
    plt.scatter(v_plot[:, 0], v_plot[:, 1], s=2, c="k", alpha=0.3, label="v samples")
    plt.axis("equal")
    plt.xlabel("v1")
    plt.ylabel("v2")
    plt.title("Velocity-space scores: KDE vs reference")
    plt.legend(loc="best")
    plt.tight_layout()

    wandb.log({"score_quiver": wandb.Image(fig_quiver)}, step=0)
    plt.show()
    plt.close(fig_quiver)

    # Main time loop with steps/sec logging
    snapshot_times = np.linspace(0.0, final_time, 6)        # [0,10,20,30,40,50]
    snapshot_steps = set(int(round(T / dt)) for T in snapshot_times)

    x_traj, v_traj, t_traj = [], [], []
    start_time = time.perf_counter()
    for istep in tqdm(range(num_steps+1)):
        if istep in snapshot_steps:
            x_host = np.asarray(x.block_until_ready())
            v_host = np.asarray(v.block_until_ready())
            x_traj.append(x_host)
            v_traj.append(v_host)
            t_traj.append(istep * dt)


        x, v, E = vlasov_step(x, v, E, cells, eta, dt, L, w)

        if C > 0:
            s = score_fn(x, v, cells, eta)
            Q = collision(x, v, s, eta, gamma, n, L, w)
            v = v - dt * C * Q

        E = E - jnp.mean(E)
        E_norm = jnp.sqrt(jnp.sum(E ** 2) * eta)
        E_L2.append(E_norm)

        # logging
        if (istep + 1) % args.log_every == 0:
            elapsed = time.perf_counter() - start_time
            steps_per_sec = (istep + 1) / elapsed
            wandb.log({
                "step": istep + 1,
                "time": float((istep + 1) * dt),
                "E_L2": float(E_norm),
                "steps_per_sec": steps_per_sec,
            }, step=istep + 1)

    # Phase-space snapshots from x_traj, v_traj
    num_snaps = len(x_traj)
    if num_snaps > 0:
        cols = min(3, num_snaps)
        rows = int(np.ceil(num_snaps / cols))

        fig_ps, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=True)
        axs = np.array(axs).reshape(-1)

        v_all = np.concatenate([np.asarray(v_snap)[:, 0] for v_snap in v_traj])
        vmin, vmax = float(v_all.min()), float(v_all.max())

        last_img = None
        for i, (x_snap, v_snap, t_snap) in enumerate(zip(x_traj, v_traj, t_traj)):
            ax = axs[i]
            xs = np.asarray(x_snap) % float(L)
            vs = np.asarray(v_snap)[:, 0]

            H, xedges, yedges, img = ax.hist2d(
                xs, vs,
                bins=[400, 400],
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
            cbar = fig_ps.colorbar(last_img, ax=axs.tolist(), orientation="vertical", fraction=0.02, pad=0.02)
            cbar.set_label("Density")

        plt.suptitle(fr"Two-stream α={alpha}, k={k}, c={c}, C={C}, n={n:.0e}, M={M}, Δt={dt}")
        plt.tight_layout()
        outdir_ps = f"data/plots/phase_space/two_stream_1d_{dv}v/"
        os.makedirs(outdir_ps, exist_ok=True)
        fname_ps = f"two_stream_phase_space_n{n:.0e}_M{M}_dt{dt}_dv{dv}_C{C}_alpha{alpha}_k{k}_c{c}.png"
        path_ps = os.path.join(outdir_ps, fname_ps)
        plt.savefig(path_ps)
        wandb.log({"phase_space_snapshots": wandb.Image(fig_ps)}, step=num_steps + 1)
        plt.show()
        plt.close(fig_ps)

    # Post-processing: two-stream growth fit (use window 10–25 as in your script)
    t_grid = jnp.linspace(0, final_time, num_steps + 1)
    t_np = np.asarray(t_grid)
    E_np = np.asarray(E_L2)

    fig_final = plt.figure(figsize=(6, 4))
    plt.plot(t_np, E_np, marker="o", ms=1, label=f"Simulation (C={C})")

    mask = (t_np > 10.0) & (t_np < 25.0)
    t_mask = t_np[mask]
    E_mask = E_np[mask]

    gamma_ref = 0.2258
    if len(t_mask) > 1:
        predicted = np.exp(gamma_ref * t_mask)
        predicted *= E_np[0] / predicted[0]
        plt.plot(
            t_mask,
            predicted,
            "r--",
            label=fr"$e^{{\gamma t}},\ \gamma = {gamma_ref:.4f}$",
        )
        wandb.log({"gamma_ref": gamma_ref})

    plt.xlabel("Time")
    plt.ylabel(r"$||E||_{L^2}$")
    plt.title(f"Two-stream: C={C}, n={n:.0e}, Δt={dt}, dv={dv}, α={alpha}, M={M}, k={k}, c={c}")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    outdir = f"data/plots/electric_field_norm/two_stream_1d_{dv}v/"
    os.makedirs(outdir, exist_ok=True)
    fname = f"two_stream_n{n:.0e}_M{M}_dt{dt}_{score_method}_dv{dv}_C{C}_alpha{alpha}_k{k}_c{c}.png"
    path = os.path.join(outdir, fname)
    plt.savefig(path)

    wandb.log({"two_stream_growth": wandb.Image(fig_final)}, step=num_steps + 2)
    wandb.save(path)

    plt.show()
    plt.close(fig_final)

    wandb_run.finish()


if __name__ == "__main__":
    main()
