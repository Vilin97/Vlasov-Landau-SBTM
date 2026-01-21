#%% Weibel instability

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
import math

import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from flax import nnx
import optax

from src import utils, score_model

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400
#%% helpers

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

    return v - jnp.mean(v, axis=0)


def weibel_score_v(v, beta, c):
    """Analytic ∇_v log f0 for the Weibel equilibrium."""
    v1 = v[:, 0]
    v2 = v[:, 1]

    s1 = -2.0 * v1 / beta

    e1 = jnp.exp(-(v2 - c) ** 2 / beta)
    e2 = jnp.exp(-(v2 + c) ** 2 / beta)
    num = (v2 - c) * e1 + (v2 + c) * e2
    den = e1 + e2
    s2 = -2.0 / beta * (num / den)

    if v.shape[1] == 2:
        return jnp.stack([s1, s2], axis=1)
    else:
        v3 = v[:, 2]
        s3 = -2.0 * v3 / beta
        return jnp.stack([s1, s2, s3], axis=1)


def lorentz_force(E1_p, E2_p, B3_p, v, dv):
    if dv == 2:
        vx, vy = v[:, 0], v[:, 1]
        Fx = E1_p + vy * B3_p
        Fy = E2_p - vx * B3_p
        return jnp.stack([Fx, Fy], axis=1)
    else:
        E_p = jnp.stack([E1_p, E2_p, jnp.zeros_like(E1_p)], axis=1)
        B_p = jnp.stack(
            [jnp.zeros_like(B3_p), jnp.zeros_like(B3_p), B3_p],
            axis=1,
        )
        return E_p + jnp.cross(v, B_p)


@jax.jit
def deposit_currents(x, v, cells, eta, w):
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
    dE2dx = (jnp.roll(E2, -1) - jnp.roll(E2, 1)) / (2.0 * eta)
    B3 = B3 - dt * dE2dx
    dB3dx = (jnp.roll(B3, -1) - jnp.roll(B3, 1)) / (2.0 * eta)
    E2 = E2 - dt * (J2 + dB3dx)
    return E2, B3


def plot_weibel(x_traj, v_traj, t_traj, dv, beta, c, k, alpha_B, n, M, dt):
    title = fr"Weibel 1D-{dv}V, β={beta}, c={c}, k={k}, α={alpha_B}, n={n:.0e}, M={M}, dt={dt}"
    num_snaps = len(x_traj)
    k_est = int(math.sqrt(x_traj[0].shape[0] / 50))
    bins = [max(20, k_est), max(20, k_est)]

    cols = min(3, num_snaps)
    rows = int(np.ceil(num_snaps / cols))

    fig, axs = plt.subplots(
        rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=True
    )
    axs = np.array(axs).reshape(-1)

    v1min, v1max = -0.7, 0.7
    v2min, v2max = -0.7, 0.7

    last_img = None
    for i, (x_snap, v_snap, t_snap) in enumerate(zip(x_traj, v_traj, t_traj)):
        ax = axs[i]
        v1s = np.asarray(v_snap)[:, 0]
        v2s = np.asarray(v_snap)[:, 1]

        H, xedges, yedges = np.histogram2d(
            v1s,
            v2s,
            bins=bins,
            range=[[v1min, v1max], [v2min, v2max]],
            density=True,
        )

        H = np.where(H > 0, H, np.nan)

        cmap = plt.cm.jet.copy()
        cmap.set_bad(color="black")

        vmin = np.nanmin(H)
        vmax = np.nanmax(H)
        img = ax.imshow(
            H.T,
            origin="lower",
            extent=[v1min, v1max, v2min, v2max],
            aspect="auto",
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax),
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
        cbar = plt.colorbar(
            last_img, ax=axs.tolist(), orientation="vertical", fraction=0.02, pad=0.02
        )
        cbar.set_label("Density")

    plt.suptitle(title)
    return fig

#%% config (defaults from CLI)

n = 10**5
M = 100
dt = 0.1
gpu = 0
fp32 = False
dv = 3
final_time = 125.0
beta = 1e-2
c = 0.3
k = 1.0 / 5.0
alpha_B = 1e-3
seed = 43

C = 0.0
score_method = "blob"

log_every = 10
num_snapshots = 6
quiver_score_scale = 500
quiver_flow_scale = 1000

#%% 

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
jax.config.update("jax_enable_x64", not fp32)
gpu_name = jax.devices()[0].device_kind

print(
    f"Weibel with n={n:.0e}, M={M}, dt={dt}, dv={dv}, "
    f"beta={beta}, c={c}, k={k}, alpha_B={alpha_B}, C={C}, "
    f"gpu={gpu}, fp32={fp32}, GPU={gpu_name}"
)

#%% initial sampling and score model

L = 2.0 * jnp.pi / k
eta = L / M
cells = (jnp.arange(M) + 0.5) * eta
w = L / n
gamma = -dv
dx = 1

key = jr.PRNGKey(seed)
key_x, key_v = jr.split(key, 2)

x = sample_x_uniform(key_x, n, L)
v = init_weibel_velocities(key_v, n, dv, beta, c)

model = None
optimizer = None
training_config = None

if score_method == "blob":
    score_fn = utils.score_blob
elif score_method == "scaled_blob":
    score_fn = utils.scaled_score_blob
elif score_method == "sbtm":
    hidden_dims = (256, 256)
    training_config = {
        "batch_size": 20_000,
        "num_epochs": 10_000,
        "abs_tol": 1e-4,
        "lr": 1e-4,
        "num_batch_steps": 100,
    }
    model = score_model.MLPScoreModel(dx, dv, hidden_dims=hidden_dims)
    example_name = "weibel"
    model_path = os.path.join(
        "data/score_models",
        f"{example_name}_dx{dx}_dv{dv}_beta{beta}_k{k}_c{c}_n{n}/hidden_{str(hidden_dims)}/epochs_{training_config['num_epochs']}",
    )
    if os.path.exists(model_path):
        model.load(model_path)
    else:
        loss_hist = utils.train_initial_model(
            model,
            x,
            v,
            weibel_score_v(v, beta, c),
            batch_size=training_config["batch_size"],
            num_epochs=training_config["num_epochs"],
            abs_tol=training_config["abs_tol"],
            lr=training_config["lr"],
            verbose=True,
            print_every=10,
        )
        try:
            model.save(model_path)
        except Exception as e:
            print(f"Warning: could not save model to {model_path}: {e}")
        time.sleep(1)
    optimizer = nnx.Optimizer(model, optax.adamw(training_config["lr"]))

    def score_fn(x_in, v_in, cells_in, eta_in):
        return model(x_in, v_in)
else:
    raise ValueError(f"Unknown score method: {score_method}")

#%% initial fields and diagnostics

E1 = jnp.zeros(M)
E2 = jnp.zeros(M)
B3 = alpha_B * jnp.sin(k * cells)

rho = utils.evaluate_charge_density(x, cells, eta, w)

def spatial_density(x_pos):
    return jnp.ones_like(x_pos) / L

def v2_target(v2):
    sigma = jnp.sqrt(beta / 2.0)
    return 0.5 * (
        jax.scipy.stats.norm.pdf(v2, c, sigma)
        + jax.scipy.stats.norm.pdf(v2, -c, sigma)
    )

fig_init = utils.visualize_initial(
    x, v[:, 1], cells, E1, rho, eta, L, spatial_density, v2_target
)
plt.close(fig_init)

if score_method == "sbtm":
    s_plot = model(x, v)
else:
    s_plot = score_fn(x, v, cells, eta)
s_true = weibel_score_v(v, beta, c)

#%%
fig_quiver = utils.plot_score_quiver(
    v, s_plot, s_true, label=score_method, scale=quiver_score_scale
)
plt.show()
plt.close(fig_quiver)

Q0_pred = utils.collision(x, v, s_plot, eta, gamma, L, w)
Q0_true = utils.collision(x, v, s_true, eta, gamma, L, w)

fig_flow0 = utils.plot_U_quiver_pred(
    v, -Q0_pred, label=f"{score_method}, t=0.0",
    U_true=-Q0_true, scale=quiver_flow_scale,
)
plt.show()
plt.close(fig_flow0)

#%% time stepping loop (good cell to tweak for quiver debugging)

final_steps = int(final_time / dt)
snapshot_times = np.linspace(0.0, final_time, num_snapshots)
snapshot_steps = set(int(round(T / dt)) for T in snapshot_times)

x_traj, v_traj, t_traj = [], [], []

start_time = time.perf_counter()
for istep in tqdm(range(final_steps + 1)):
    if istep in snapshot_steps:
        x_traj.append(np.asarray(x.block_until_ready()))
        v_traj.append(np.asarray(v.block_until_ready()))
        t_traj.append(istep * dt)

        if score_method == "sbtm":
            s_snap = model(x, v)
        else:
            s_snap = score_fn(x, v, cells, eta)
        Q_snap = utils.collision(x, v, s_snap, eta, gamma, L, w)

        fig_quiver_score_snap = utils.plot_score_quiver_pred(
            v, s_snap,
            label=f"{score_method}, t={istep * dt:.2f}",
            scale=quiver_score_scale,
        )
        plt.show()
        plt.close(fig_quiver_score_snap)

        fig_quiver_flow_snap = utils.plot_U_quiver_pred(
            v, -Q_snap,
            label=f"{score_method}, t={istep * dt:.2f}",
            scale=quiver_flow_scale,
        )
        plt.show()
        plt.close(fig_quiver_flow_snap)

    E1_p = utils.evaluate_field_at_particles(E1, x, cells, eta)
    E2_p = utils.evaluate_field_at_particles(E2, x, cells, eta)
    B3_p = utils.evaluate_field_at_particles(B3, x, cells, eta)

    F = lorentz_force(E1_p, E2_p, B3_p, v, dv)
    v = v + dt * F
    x = jnp.mod(x + dt * v[:, 0], L)

    if C > 0.0:
        if score_method == "sbtm":
            key_train = jr.PRNGKey(istep)
            utils.train_score_model(
                model,
                optimizer,
                x,
                v,
                key_train,
                batch_size=training_config["batch_size"],
                num_batch_steps=training_config["num_batch_steps"],
            )
            s = model(x, v)
        else:
            s = score_fn(x, v, cells, eta)
        Q = utils.collision(x, v, s, eta, gamma, L, w)
        v = v - dt * C * Q
        entropy_production = jnp.mean(jnp.sum(s * C * Q, axis=1))
    else:
        entropy_production = 0.0

    E1 = utils.update_electric_field(E1, x, v, cells, eta, w, dt)

    _, J2 = deposit_currents(x, v, cells, eta, w)
    E2, B3 = maxwell_step(E2, B3, J2, eta, dt)

    kinetic_energy = 0.5 * jnp.mean(jnp.sum(v**2, axis=1))
    E1_energy = 0.5 * jnp.sum(E1**2) * eta
    E2_energy = 0.5 * jnp.sum(E2**2) * eta
    magnetic_energy = 0.5 * jnp.sum(B3**2) * eta
    electric_energy = E1_energy + E2_energy
    total_energy = kinetic_energy + electric_energy + magnetic_energy

    E_L2 = jnp.sqrt(jnp.sum(E1**2 + E2**2) * eta)

    if (istep + 1) % log_every == 0:
        elapsed = time.perf_counter() - start_time
        steps_per_sec = (istep + 1) / elapsed

#%% final phase-space snapshots

fig_ps = plot_weibel(x_traj, v_traj, t_traj, dv, beta, c, k, alpha_B, n, M, dt)
plt.show()
plt.close(fig_ps)


# %%
