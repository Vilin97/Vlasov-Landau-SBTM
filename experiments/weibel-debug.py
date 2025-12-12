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

# ----------------- params -----------------
n = 10**6          # must be even
M = 100
dt = 0.05
gpu = 0
fp32 = False
dv = 2            # 1d-2v for Weibel
final_time = 50.0
alpha = 0.0       # uniform in x
k = 1/5
c = 2.4           # beam shift in v_y
C = 0.0           # collisionless
seed = 43
print(f"Using seed {seed}")

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
jax.config.update("jax_enable_x64", not fp32)

# ----------------- helpers -----------------
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
    k = jnp.argmax(var_v)
    x_star = 0.5 * (edges[k] + edges[k + 1])
    return x_star, edges, var_v

def init_weibel_velocities(key_v, n, c, vth_par=1.0, vth_perp=1.0):
    assert n % 2 == 0
    n_half = n // 2
    k1, k2, k3 = jr.split(key_v, 3)
    vx = jr.normal(k1, (n,)) * vth_par
    vy1 = jr.normal(k2, (n_half,)) * vth_perp + c
    vy2 = jr.normal(k3, (n_half,)) * vth_perp - c
    vy = jnp.concatenate([vy1, vy2])
    v = jnp.stack([vx, vy], axis=1)
    v = v - jnp.mean(v, axis=0)
    return v

def sample_x(key_x, n, M, alpha, k):
    assert n % 2 == 0
    L = 2 * jnp.pi / k
    if alpha == 0.0:
        return jr.uniform(key_x, (n,), minval=0.0, maxval=L)

    x_edges = jnp.linspace(0.0, L, M + 1)
    widths = jnp.diff(x_edges)
    cell_integrals = widths + (alpha / k) * (
        jnp.sin(k * x_edges[1:]) - jnp.sin(k * x_edges[:-1])
    )
    n_half = n // 2
    counts_float = cell_integrals / jnp.sum(cell_integrals) * n_half
    counts_floor = jnp.floor(counts_float).astype(jnp.int32)
    remainder = int(n_half - jnp.sum(counts_floor))
    frac = counts_float - counts_floor
    idx_sorted = jnp.argsort(frac)
    counts = jnp.where(
        jnp.isin(jnp.arange(M), idx_sorted[-remainder:]),
        counts_floor + 1,
        counts_floor,
    )
    cell_ids = jnp.repeat(jnp.arange(M, dtype=jnp.int32), counts)
    key_u, _ = jr.split(key_x)
    u = jr.uniform(key_u, (n_half,))
    x_left = x_edges[:-1][cell_ids]
    widths_cells = widths[cell_ids]
    x_half = x_left + widths_cells * u
    x_full = jnp.concatenate([x_half, L - x_half])
    return x_full

def v_target(vv):
    return 0.5 * (
        jax.scipy.stats.norm.pdf(vv, -c, 1.0)
        + jax.scipy.stats.norm.pdf(vv, c, 1.0)
    )

# ----------------- domain, init -----------------
L = 2 * jnp.pi / k
eta = L / M
cells = (jnp.arange(M) + 0.5) * eta
w = L / n

key = jr.PRNGKey(seed)
key_x, key_v, _ = jr.split(key, 3)

x = sample_x(key_x, n, M, alpha, k)
v = init_weibel_velocities(key_v, n, c)

rho = utils.evaluate_charge_density(x, cells, eta, w)
E1 = jnp.cumsum(rho - 1) * eta    # longitudinal E
E2 = jnp.zeros_like(E1)           # transverse E_y
B3 = 1e-3 * jnp.sin(k * cells)    # small initial B_z

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
        # print(f"Step {istep}, Time {istep*dt:.2f}, vortex x ~ {vl:.4f}")

    # fields at particles
    E1_p = utils.evaluate_field_at_particles(E1, x, cells, eta)
    E2_p = utils.evaluate_field_at_particles(E2, x, cells, eta)
    B3_p = utils.evaluate_field_at_particles(B3, x, cells, eta)

    # Lorentz force (q = 1, c = 1), v = (vx, vy, 0), B = (0, 0, B3)
    vx, vy = v[:, 0], v[:, 1]
    Fx = E1_p + vy * B3_p
    Fy = E2_p - vx * B3_p

    v = v.at[:, 0].add(dt * Fx)
    v = v.at[:, 1].add(dt * Fy)

    x = jnp.mod(x + dt * v[:, 0], L)

    # update E1 using existing electrostatic Ampere-like update
    E1 = utils.update_electric_field(E1, x, v, cells, eta, w, dt)

    # deposit transverse current and update (E2,B3)
    J1, J2 = utils.deposit_currents(x, v, cells, eta, w)
    E2, B3 = utils.maxwell_step(E2, B3, J2, eta, dt)

title = fr"Weibel 1d-2v α={alpha}, k={k}, c={c}, C=0, n={n:.0e}, M={M}, dt={dt}"
fig_ps, _ = utils.plot_phase_space_snapshots(x_traj, v_traj, t_traj, L, title, save=False)
plt.show()
plt.close(fig_ps)

# %%
