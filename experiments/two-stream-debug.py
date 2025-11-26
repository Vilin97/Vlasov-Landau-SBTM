#%% imports
import os
import time

import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src import utils

for seed in range(40,50):
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
    C = 0.0            # collisionless
    # seed = 50 # with dv=2 seed 46 gives a very shifted vortex. seed 42 works well.
    print(f"Using seed {seed}")

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    jax.config.update("jax_enable_x64", not fp32)

    def vortex_location(x, v, nbins=50, min_count=10):
        """Estimate the location x where Var[v|x] is maximized."""
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

        k = jnp.argmax(var_v)
        x_star = 0.5 * (edges[k] + edges[k + 1])
        return x_star, edges, var_v

    def init_two_stream_velocities(key_v, n, dv, c):
        assert n % 2 == 0
        n_half = n // 2
        key_v1, key_v2 = jr.split(key_v)
        v1 = jr.normal(key_v1, (n_half, dv))
        v1 = v1.at[:, 0].add(c)
        v2 = -v1
        v = jnp.vstack([v1, v2])
        # v2 = jr.normal(key_v2, (n_half, dv))
        # v2 = v2.at[:, 0].add(-c)
        # v = jnp.vstack([v1, v2])
        # v -= jnp.mean(v, axis=0)
        return v


    L = 2 * jnp.pi / k
    eta = L / M
    cells = (jnp.arange(M) + 0.5) * eta
    w = L / n

    def spatial_density(x):
        return (1 + alpha * jnp.cos(k * x)) / (2 * jnp.pi / k)

    max_value = jnp.max(spatial_density(cells))

    key = jr.PRNGKey(seed)
    key_x, key_v, perm_key = jr.split(key, 3)
    domain = (0.0, L)
    x1 = utils.rejection_sample(key_x, spatial_density, domain, max_value=max_value, num_samples=n//2) 
    x2 = L - x1 
    x = jnp.concatenate([x1, x2])

    v = init_two_stream_velocities(key_v, n, dv, c)

    shuffle_keys = jr.split(perm_key, 2)
    perm1 = jr.permutation(shuffle_keys[0], n)
    perm2 = jr.permutation(shuffle_keys[1], n)
    x = x[perm1]
    v = v[perm2]

    rho = utils.evaluate_charge_density(x, cells, eta, w)
    E = jnp.cumsum(rho - 1) * eta

    def v_target(vv):
        return 0.5 * (jax.scipy.stats.norm.pdf(vv, -c, 1.0) + jax.scipy.stats.norm.pdf(vv, c, 1.0))

    fig_init = utils.visualize_initial(x, v[:, 0], cells, E, rho, eta, L, spatial_density, v_target)
    plt.show()
    plt.close(fig_init)

    final_steps = int(final_time / dt)

    snapshot_times = np.linspace(0.0, final_time, 6)
    snapshot_steps = set(int(round(T / dt)) for T in snapshot_times)

    x_traj, v_traj, t_traj = [], [], []
    for istep in tqdm(range(final_steps + 1)):
        # snapshots
        if istep in snapshot_steps:
            x_traj.append(np.asarray(x.block_until_ready()))
            v_traj.append(np.asarray(v.block_until_ready()))
            t_traj.append(istep * dt)
            vl,_,_ = vortex_location(x, v[:, 0])
            print(f"Step {istep}, Time {istep*dt:.2f}, Vortex at x={vl:.4f}, Vortex shift = {vl - L/2:.4f}")
            # print(v.mean(axis=0))
            # print(E.mean())

        E_at_particles = utils.evaluate_field_at_particles(E, x, cells, eta)
        v = v.at[:, 0].add(dt * E_at_particles)
        x = jnp.mod(x + dt * v[:, 0], L)
        E = utils.update_electric_field(E, x, v, cells, eta, w, dt)

    title = fr"Two-stream dv={dv} α={alpha}, k={k}, c={c}, C=0, n={n:.0e}, M={M}, dt={dt}"
    fig_ps, _ = utils.plot_phase_space_snapshots(x_traj, v_traj, t_traj, L, title, save=False)
    for ax in fig_ps.axes[:-1]:
        ax.text(0.5, 0.5, 'x', color='red', fontsize=18, ha='center', va='center', transform=ax.transAxes)
    plt.show()
    plt.close(fig_ps)

    # %%
