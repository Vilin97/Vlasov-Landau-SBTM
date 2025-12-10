#%% imports
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time

import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src import utils

# --------- TSI sampling helpers ---------
def q_fcn(v, q_std):
    return (1 / (q_std * np.sqrt(2 * np.pi))) * np.exp(-(v**2) / (2 * q_std**2))

def f_v(v, v_bar):
    return (1 / (2 * np.sqrt(2 * np.pi))) * (
        np.exp(-((v - v_bar) ** 2) / 2) + np.exp(-((v + v_bar) ** 2) / 2)
    )

def AR_sampling_TSI_v(num_samples, f_v_init, v_bar, q, q_std, A):
    samples_v = []
    while len(samples_v) < num_samples:
        u = np.random.uniform(0, 1, num_samples)
        v = np.random.normal(0, q_std, num_samples)
        pi_val = f_v_init(v, v_bar)
        q_val = q(v, q_std)

        accept = u * A * q_val <= pi_val
        samples_v.extend(v[accept])
        samples_v = samples_v[:num_samples]

    return np.array(samples_v)

def rand_sampling_TSI(
    num_grid_x, tot_ptc, alpha, k, v_bar, randomseed,
    f_v=f_v, q=q_fcn, q_std=3, A=2.5
):
    np.random.seed(randomseed)
    x_min, x_max = 0, 2 * np.pi / k
    L_x = x_max - x_min
    x_edges = np.linspace(x_min, x_max, num_grid_x + 1)
    x_widths = np.diff(x_edges)

    cell_integrals = x_widths + (alpha / k) * (
        np.sin(k * x_edges[1:]) - np.sin(k * x_edges[:-1])
    )

    counts_float = cell_integrals / np.sum(cell_integrals) * (tot_ptc // 2)
    counts_floor = np.floor(counts_float).astype(int)
    remainder = (tot_ptc // 2) - np.sum(counts_floor)

    frac = counts_float - counts_floor
    counts_floor[np.argsort(frac)[-remainder:]] += 1
    particle_counts = counts_floor

    x_samples_half = []
    for i in range(num_grid_x):
        x0, x1 = x_edges[i], x_edges[i + 1]
        x_cell = np.random.uniform(x0, x1, particle_counts[i])
        x_samples_half.append(x_cell)
    samples_x_half = np.concatenate(x_samples_half)

    samples_x_half2 = L_x - samples_x_half
    samples_x = np.concatenate((samples_x_half, samples_x_half2))

    v_samples_half = AR_sampling_TSI_v(
        tot_ptc // 2, f_v, v_bar, q, q_std, A
    )
    samples_v = np.concatenate((v_samples_half, -v_samples_half))

    weights = np.full(tot_ptc, (x_max - x_min) / tot_ptc)

    return {
        "samples_x": samples_x,
        "samples_v": samples_v,
        "weights": weights,
    }

# --------- original script params ---------
n = 10**6          # must be even
M = 100
dt = 0.05
gpu = 0
fp32 = False
dv = 1
final_time = 50.0
alpha = 1/200
k = 1/5
c = 2.4
C = 0.0
# seed = 43
for seed in range(50):
    print(f"Using seed {seed}")

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    jax.config.update("jax_enable_x64", not fp32)

    def vortex_location(x, v, nbins=200, min_count=10):
        x_min, x_max = jnp.min(x), jnp.max(x)
        edges = jnp.linspace(x_min, x_max, nbins + 1)

        idx = jnp.searchsorted(edges, x, side="right") - 1
        idx = jnp.clip(idx, 0, nbins - 1)

        counts = jnp.bincount(idx, length=nbins)
        sum_v  = jnp.bincount(idx, weights=v,      length=nbins)
        sum_v2 = jnp.bincount(idx, weights=v * v,  length=nbins)

        mean_v = sum_v / jnp.maximum(counts, 1)
        var_v  = sum_v2 / jnp.maximum(counts, 1) - mean_v ** 2
        var_v  = jnp.where(counts >= min_count, var_v, -jnp.inf)

        k_bin = jnp.argmax(var_v)
        x_star = 0.5 * (edges[k_bin] + edges[k_bin + 1])
        return x_star, edges, var_v

    L = 2 * jnp.pi / k
    eta = L / M
    cells = (jnp.arange(M) + 0.5) * eta
    w = L / n

    def spatial_density(x):
        return (1 + alpha * jnp.cos(k * x)) / (2 * jnp.pi / k)

    def v_target(vv):
        return 0.5 * (
            jax.scipy.stats.norm.pdf(vv, -c, 1.0)
            + jax.scipy.stats.norm.pdf(vv,  c, 1.0)
        )

    # --------- NEW: TSI sampling with explicit random seed ---------
    assert n % 2 == 0
    assert dv == 1

    samples = rand_sampling_TSI(
        num_grid_x=M,
        tot_ptc=n,
        alpha=alpha,
        k=k,
        v_bar=c,
        randomseed=seed,
        f_v=f_v,
        q=q_fcn,
        q_std=3,
        A=2.5,
    )

    x = jnp.array(samples["samples_x"])
    v = jnp.array(samples["samples_v"]).reshape(n, dv)

    rho = utils.evaluate_charge_density(x, cells, eta, w)
    E = jnp.cumsum(rho - 1) * eta

    fig_init = utils.visualize_initial(
        x, v[:, 0], cells, E, rho, eta, L, spatial_density, v_target
    )
    plt.show()
    plt.close(fig_init)

    final_steps = int(final_time / dt)
    snapshot_times = np.linspace(0.0, final_time, 6)
    snapshot_steps = set(int(round(T / dt)) for T in snapshot_times)

    x_traj, v_traj, t_traj = [], [], []
    for istep in tqdm(range(final_steps + 1)):
        if istep in snapshot_steps:
            x_traj.append(np.asarray(x.block_until_ready()))
            v_traj.append(np.asarray(v.block_until_ready()))
            t_traj.append(istep * dt)
            vl, _, _ = vortex_location(x, v[:, 0])
            print(
                f"Step {istep}, Time {istep*dt:.2f}, "
                f"Vortex at x={vl:.4f}, Vortex shift = {vl - L/2:.4f}"
            )

        E_at_particles = utils.evaluate_field_at_particles(E, x, cells, eta)
        v = v.at[:, 0].add(dt * E_at_particles)
        x = jnp.mod(x + dt * v[:, 0], L)
        E = utils.update_electric_field(E, x, v, cells, eta, w, dt)

    title = (
        fr"Two-stream dv={dv} α={alpha}, k={k}, c={c}, C=0, "
        fr"n={n:.0e}, M={M}, dt={dt}"
    )
    fig_ps, _ = utils.plot_phase_space_snapshots(
        x_traj, v_traj, t_traj, L, title, save=False
    )
    for ax in fig_ps.axes[:-1]:
        ax.text(
            0.5,
            0.5,
            "x",
            color="red",
            fontsize=18,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    plt.show()
    plt.close(fig_ps)

    # %%
