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

#%% parameters (formerly defaults)
n = 10**6          # must be even
M = 100
dt = 0.05
gpu = 0
fp32 = False
dv = 2
final_time = 50.0
alpha = 1/200
k = 1/5
c = 2.4
C = 0.0            # collisionless
seed = 42

#%% environment
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
jax.config.update("jax_enable_x64", not fp32)

#%% init two-stream velocities
def init_two_stream_velocities(key_v, n, dv, c):
    assert n % 2 == 0
    n_half = n // 2
    v1 = jr.normal(key_v, (n_half, dv))
    v1 -= jnp.mean(v1, axis=0)
    v1 = v1.at[:, 0].add(c)
    v2 = -v1
    return jnp.vstack([v1, v2])

key_v, key_x = jr.split(jr.PRNGKey(seed), 2)
v = init_two_stream_velocities(key_v, n, dv, c)

#%% initialize positions via rejection sampling
L = 2 * jnp.pi / k
eta = L / M
cells = (jnp.arange(M) + 0.5) * eta
w = L / n   # q=1

def spatial_density(x):
    return (1 + alpha * jnp.cos(k * x)) / (2 * jnp.pi / k)

max_value = jnp.max(spatial_density(cells))
domain = (0.0, float(L))
x = utils.rejection_sample(key_x, spatial_density, domain, max_value=max_value, num_samples=n)

#%% initial E-field
rho = utils.evaluate_charge_density(x, cells, eta, w)
E = jnp.cumsum(rho - 1) * eta
E = E - jnp.mean(E)

#%% plot initial
def v_target(vv):
    return 0.5 * (jax.scipy.stats.norm.pdf(vv, -c, 1.0) + jax.scipy.stats.norm.pdf(vv, c, 1.0))

fig_init = utils.visualize_initial(x, v[:, 0], cells, E, rho, eta, L, spatial_density, v_target)
plt.show()
plt.close(fig_init)

#%% time stepping
final_steps = int(final_time / dt)
E_L2 = [jnp.sqrt(jnp.sum(E**2) * eta)]

snapshot_times = np.linspace(0.0, final_time, 6)
snapshot_steps = set(int(round(T / dt)) for T in snapshot_times)

x_traj, v_traj, t_traj = [], [], []
start = time.perf_counter()

for istep in tqdm(range(final_steps + 1)):
    # snapshots
    if istep in snapshot_steps:
        x_traj.append(np.asarray(x.block_until_ready()))
        v_traj.append(np.asarray(v.block_until_ready()))
        t_traj.append(istep * dt)
        print(x.mean() - L/2)
        print(v.mean(axis=0))

    # collisionless Vlasov step
    x, v, E = utils.vlasov_step(x, v, E, cells, eta, dt, L, w)
    E = E - jnp.mean(E)

#%% phase-space snapshot plots
title = fr"Two-stream α={alpha}, k={k}, c={c}, C=0, n={n:.0e}, M={M}, dt={dt}"
fig_ps, _ = utils.plot_phase_space_snapshots(x_traj, v_traj, t_traj, L, title, save=False)
plt.show()
plt.close(fig_ps)


# %%
