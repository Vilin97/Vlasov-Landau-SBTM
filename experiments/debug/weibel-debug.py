#%%

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import math
from tqdm import tqdm
from src import utils

@jax.jit
def deposit_currents(x, v, cells, eta, w):
    """J1,J2 on grid from particles (linear-hat, periodic)."""
    M = cells.size
    idx_f = x / eta - 0.5
    i0 = jnp.floor(idx_f).astype(jnp.int32) % M
    i1 = (i0 + 1) % M
    f = idx_f - jnp.floor(idx_f)
    w0, w1 = 1.0 - f, f

    J1 = (
        jnp.zeros(M)
        .at[i0].add(w0 * v[:, 0])
        .at[i1].add(w1 * v[:, 0])
    )
    J2 = (
        jnp.zeros(M)
        .at[i0].add(w0 * v[:, 1])
        .at[i1].add(w1 * v[:, 1])
    )
    scale = w / eta
    return scale * J1, scale * J2

@jax.jit
def maxwell_step(E2, B3, J2, eta, dt):
    """1D Maxwell for transverse fields."""
    dE2dx = (jnp.roll(E2, -1) - jnp.roll(E2, 1)) / (2.0 * eta)
    B3 = B3 - dt * dE2dx
    dB3dx = (jnp.roll(B3, -1) - jnp.roll(B3, 1)) / (2.0 * eta)
    E2 = E2 - dt * (J2 + dB3dx)
    return E2, B3


# ---------- parameters from paper ----------
n = 10**6
M = 100
dt = 0.1
final_time = 125
gpu = 0
fp32 = False
dv = 2
beta = 1e-2
c = 0.3
k = 1.0 / 5.0
alpha_B = 1e-3       # B3 amplitude
seed = 43

print(f"dv={dv}, n={n}, M={M}, dt={dt}, T={final_time}")
assert dv in (2, 3)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
jax.config.update("jax_enable_x64", not fp32)

# ---------- helpers ----------
def sample_x_uniform(key_x, n, L):
    return jr.uniform(key_x, (n,), minval=0.0, maxval=L)

def init_weibel_velocities(key_v, n, dv, beta, c):
    """f0(v1,v2) ∝ exp(-v1^2/β)[exp(-(v2-c)^2/β)+exp(-(v2+c)^2/β)]."""
    assert n % 2 == 0
    n_half = n // 2
    sigma = jnp.sqrt(beta / 2.0)

    k1, k2, k3, k4 = jr.split(key_v, 4)
    v1 = jr.normal(k1, (n,)) * sigma

    v2_1 = jr.normal(k2, (n_half,)) * sigma + c
    v2_2 = jr.normal(k3, (n_half,)) * sigma - c
    v2 = jnp.concatenate([v2_1, v2_2])

    if dv == 2:
        v = jnp.stack([v1, v2], axis=1)
    else:
        v3 = jr.normal(k4, (n,)) * sigma
        v = jnp.stack([v1, v2, v3], axis=1)

    return v - jnp.mean(v, axis=0)  # zero net momentum

def lorentz_force(E1_p, E2_p, B3_p, v, dv):
    if dv == 2:
        vx, vy = v[:, 0], v[:, 1]
        Fx = E1_p + vy * B3_p
        Fy = E2_p - vx * B3_p
        return jnp.stack([Fx, Fy], axis=1)
    else:
        E_p = jnp.stack([E1_p, E2_p, jnp.zeros_like(E1_p)], axis=1)
        B_p = jnp.stack([
            jnp.zeros_like(B3_p),
            jnp.zeros_like(B3_p),
            B3_p,
        ], axis=1)
        return E_p + jnp.cross(v, B_p)

# ---------- domain and init ----------
L = 2.0 * jnp.pi / k          # x ∈ (0, 2π/k)
eta = L / M
cells = (jnp.arange(M) + 0.5) * eta
w = L / n

key = jr.PRNGKey(seed)
key_x, key_v = jr.split(key, 2)

x = sample_x_uniform(key_x, n, L)
v = init_weibel_velocities(key_v, n, dv, beta, c)

# electric field initialised self-consistently to zero
E1 = jnp.zeros(M)
E2 = jnp.zeros(M)

# B3(0,x) = α sin(kx)
B3 = alpha_B * jnp.sin(k * cells)

#%%
# plot the v1-v2 marginal
bounds_v = [(-0.7, 0.7)] * 2
bins_per_side = 200
density_vals = utils.density_on_regular_grid(v[:, :2], bounds_v=bounds_v, bins_per_side=bins_per_side, x=None, bounds_x=None, smooth_sigma_bins=0.0)
fig, ax = plt.subplots(figsize=(6, 5))

H = density_vals
H = np.where(H > 0, H, np.nan)

cmap = plt.cm.jet.copy()
cmap.set_bad(color="black")

v1min, v1max = bounds_v[0][0], bounds_v[0][1]
v2min, v2max = bounds_v[1][0], bounds_v[1][1]

vmin = np.nanmin(H)
vmax = np.nanmax(H)
im = ax.imshow(
    H.T,
    origin="lower",
    extent=[v1min, v1max, v2min, v2max],
    aspect="auto",
    cmap=cmap,
    norm=LogNorm(vmin=vmin, vmax=vmax),
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Density")
ax.set_xlabel("v1")
ax.set_ylabel("v2")
ax.set_title(f"Initial velocity distribution (dv={dv})")
plt.show()

#%%
# plot the v1-v2 slice at x~=π/16
bounds_v = [(-0.7, 0.7)] * 2
bounds_x = (0.0625*math.pi-0.05, 0.0625*math.pi+0.05)
bins_per_side = (1, 200, 200)
density_vals = utils.density_on_regular_grid(v[:, :2], bounds_v=bounds_v, bins_per_side=bins_per_side, x=x, bounds_x=bounds_x, smooth_sigma_bins=0.0)
fig, ax = plt.subplots(figsize=(6, 5))

H = density_vals
H = np.where(H > 0, H, np.nan)

cmap = plt.cm.jet.copy()
cmap.set_bad(color="black")

v1min, v1max = bounds_v[0][0], bounds_v[0][1]
v2min, v2max = bounds_v[1][0], bounds_v[1][1]

vmin = np.nanmin(H)
vmax = np.nanmax(H)
im = ax.imshow(
    H.T,
    origin="lower",
    extent=[v1min, v1max, v2min, v2max],
    aspect="auto",
    cmap=cmap,
    norm=LogNorm(vmin=vmin, vmax=vmax),
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Density")
ax.set_xlabel("v1")
ax.set_ylabel("v2")
ax.set_title(f"Initial velocity distribution (dv={dv})")
plt.show()

#%%
# plot the v1~=0 slice of the v1-v2 marginal
bounds_v = [(-0.01, 0.01), (-3, 3)]
bins_per_side = (1, 200)
density_vals = utils.density_on_regular_grid(v[:,:2], bounds_v=bounds_v, bins_per_side=bins_per_side, x=None, bounds_x=None, smooth_sigma_bins=0.0)

v1_grid = np.linspace(bounds_v[0][0], bounds_v[0][1], bins_per_side[0])
v2_grid = np.linspace(bounds_v[1][0], bounds_v[1][1], bins_per_side[1])

# Extract density at v1=0 and plot as a line
v1_idx = np.argmin(np.abs(v1_grid))

# Extract the slice at v1=0
density_at_v1_0 = density_vals[v1_idx, :]

plt.figure(figsize=(8, 5))
plt.plot(v2_grid, density_at_v1_0, linewidth=2)

# Normalize the Gaussian
gaussian_density = np.exp(-0.5 * v2_grid**2) / np.sqrt(2 * np.pi)

plt.plot(v2_grid, gaussian_density, 'r--', linewidth=2, label='Standard Gaussian')
plt.legend()

plt.xlabel("v2")
plt.ylabel("Density")
plt.title(f"Initial velocity distribution at v1=0 (dv={dv})")
plt.grid(True, alpha=0.3)
plt.show()

#%%
# plot the v1 marginal
bounds_v = [(-3, 3)]
bins_per_side = 200
density_vals = utils.density_on_regular_grid(v[:,1:2], bounds_v=bounds_v, bins_per_side=bins_per_side, x=None, bounds_x=None, smooth_sigma_bins=0.0)

# Create v1 grid
v1_grid = np.linspace(bounds_v[0][0], bounds_v[0][1], bins_per_side)

plt.figure(figsize=(8, 5))
plt.plot(v1_grid, density_vals, linewidth=2)
# Plot standard Gaussian for comparison
gaussian_density = np.exp(-0.5 * v1_grid**2) / np.sqrt(2 * np.pi)
plt.plot(v1_grid, gaussian_density, 'r--', linewidth=2, label='Standard Gaussian')
plt.legend()

plt.xlabel("v1")
plt.ylabel("Density")
plt.title(f"Initial v1 marginal distribution (dv={dv})")
plt.grid(True, alpha=0.3)
plt.show()



#%%
# ---------- time loop ----------
final_steps = int(final_time / dt)
snapshot_times = np.linspace(0.0, final_time, 6)
snapshot_steps = set(int(round(T / dt)) for T in snapshot_times)

x_traj, v_traj, t_traj = [], [], []
for istep in tqdm(range(final_steps + 1)):
    if istep in snapshot_steps:
        x_traj.append(np.asarray(x.block_until_ready()))
        v_traj.append(np.asarray(v.block_until_ready()))
        t_traj.append(istep * dt)

    E1_p = utils.evaluate_field_at_particles(E1, x, cells, eta)
    E2_p = utils.evaluate_field_at_particles(E2, x, cells, eta)
    B3_p = utils.evaluate_field_at_particles(B3, x, cells, eta)

    F = lorentz_force(E1_p, E2_p, B3_p, v, dv)
    v = v + dt * F
    x = jnp.mod(x + dt * v[:, 0], L)

    # longitudinal field from J1 (same PIC machinery)
    E1 = utils.update_electric_field(E1, x, v, cells, eta, w, dt)

    # transverse fields from J2
    _, J2 = deposit_currents(x, v, cells, eta, w)
    E2, B3 = maxwell_step(E2, B3, J2, eta, dt)

# %%
def plot(x_traj, v_traj, t_traj, dv, beta, c, k, alpha_B, n, M, dt):
    title = fr"Weibel 1D-{dv}V, β={beta}, c={c}, k={k}, α={alpha_B}, n={n:.0e}, M={M}, dt={dt}"
    num_snaps = len(x_traj)
    k = int(math.sqrt(x_traj[0].shape[0] / 50))
    bins = [max(20, k), max(20, k)]

    cols = min(3, num_snaps)
    rows = int(np.ceil(num_snaps / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                            sharex=True, sharey=True)
    axs = np.array(axs).reshape(-1)

    v1_all = np.concatenate([np.asarray(v_snap)[:, 0] for v_snap in v_traj])
    v2_all = np.concatenate([np.asarray(v_snap)[:, 1] for v_snap in v_traj])
    v1min, v1max = -0.7, 0.7
    v2min, v2max = -0.7, 0.7

    last_img = None
    for i, (x_snap, v_snap, t_snap) in enumerate(zip(x_traj, v_traj, t_traj)):
        ax = axs[i]
        v1s = np.asarray(v_snap)[:, 0]
        v2s = np.asarray(v_snap)[:, 1]

        H, xedges, yedges = np.histogram2d(
            v1s, v2s,
            bins=bins,
            range=[[v1min, v1max], [v2min, v2max]],
            density=True,
        )

        # Mark zero-density bins as NaN so they appear as “background”
        H = np.where(H > 0, H, np.nan)

        # Create a colormap with NaN → black
        cmap = plt.cm.jet.copy()
        cmap.set_bad(color="black")

        img = ax.imshow(
            H.T,
            origin="lower",
            extent=[v1min, v1max, v2min, v2max],
            aspect="auto",
            cmap=cmap,
            norm=LogNorm(vmin=np.nanmin(H), vmax=np.nanmax(H)),
        )
        last_img = img

        ax.set_title(f"t = {t_snap:.1f}")
        ax.set_xlim(v1min, v1max)
        ax.set_ylim(v2min, v2max)
        ax.set_xlabel("v1")
        ax.set_ylabel("v2")

    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    if last_img is not None:
        cbar = fig.colorbar(last_img, ax=axs.tolist(),
                            orientation="vertical", fraction=0.02, pad=0.02)
        cbar.set_label("Density")

    plt.suptitle(title)
    plt.show()

plot(x_traj, v_traj, t_traj, dv, beta, c, k, alpha_B, n, M, dt)
# %%
