#%%
import os
import time

import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from src import path, utils, loss

from flax import nnx
import optax
from src import score_model

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
jax.config.update("jax_enable_x64", True)

# Parameters
seed = 42
q = 1
dx = 1
dv = 2
alpha = 0.1
k = 0.5
L = 2 * jnp.pi / k
n = 1000_000
M = 100
dt = 0.02
eta = L / M
cells = (jnp.arange(M) + 0.5) * eta
w = q * L / n
C = 0.1
gamma = -dv

key_v, key_x = jr.split(jr.PRNGKey(seed), 2)
v = jr.normal(key_v, (n, dv))
v = v - jnp.mean(v, axis=0)

def spatial_density(x):
        return (1 + alpha * jnp.cos(k * x)) / (2 * jnp.pi / k)

max_value = jnp.max(spatial_density(cells))
domain = (0.0, float(L))
x = utils.rejection_sample(key_x, spatial_density, domain, max_value=max_value, num_samples=n)

#%%

# -------------------- losses --------------------
def mse_loss(model, batch):
    x, v, s = batch
    pred = model(x, v)
    return loss.mse(pred, s)

def ism_loss(model, batch, key):
    x, v = batch
    return loss.implicit_score_matching_loss(model, x, v, key=key)

# -------------------- single steps --------------------
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

# -------------------- training loops --------------------
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
            batch = (x_sh[i:i+batch_size],
                     v_sh[i:i+batch_size],
                     s_sh[i:i+batch_size])
            supervised_step(model, optimizer, batch)

def train_score_model(model, optimizer, x, v, key, batch_size, num_batch_steps, **kwargs):
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

#%%
# Plan for sbtm:
# 1. Train the initial NN
# 2. Save the model
# 3. Load the model
# 4. Use the model in the time stepping loop
# 5. Finetune the model at each time stepping step

hidden_dims = (256,256)
model = score_model.MLPScoreModel(dx, dv, hidden_dims=hidden_dims)
# model = score_model.ResNetScoreModel(dx, dv, hidden_dims=hidden_dims)
score_method = "sbtm"


# Define training configuration
training_config = {
    "batch_size": 2000,
    "num_epochs": 1000, # initial training
    "abs_tol": 1e-3,
    "lr": 2e-4,
    "num_batch_steps": 10,  # at each step
    # "train_every": 1   # train every k time steps
}

"Train and save the initial model"
example_name = "landau_damping"
model_path = os.path.join(path.MODELS, f'{example_name}_dx{dx}_dv{dv}_alpha{alpha}_k{k}/hidden_{str(hidden_dims)}/epochs_{training_config["num_epochs"]}')
if not os.path.exists(model_path):
    train_initial_model(model, x, v, -v, verbose=True, **training_config)
    model.save(model_path)
time.sleep(0.1)  # wait for the model to be saved
model.load(model_path)

#%%
rho = utils.evaluate_charge_density(x, cells, eta, w)
E = jnp.cumsum(rho - 1) * eta
E = E - jnp.mean(E)

final_time = 15.0
num_steps = int(final_time / dt)
t = 0.0
E_L2 = [jnp.sqrt(jnp.sum(E ** 2) * eta)]

print(
    f"Landau Damping with n={n:.0e}, M={M}, dt={dt}, eta={float(eta):.4f}"
)

# Quiver of scores before time stepping
s_predicted = model(x, v)
s_true = -v

fig_quiver = utils.plot_score_quiver(v, s_predicted, s_true, label="sbtm")

plt.show()
plt.close(fig_quiver)

#%%
# Main time loop with steps/sec logging
optimizer = nnx.Optimizer(model, optax.adamw(training_config["lr"]))
snapshot_times = np.linspace(0.0, final_time, 6)
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

    x, v, E = utils.vlasov_step(x, v, E, cells, eta, dt, L, w)

    if C>0:
        s = model(x, v)
        Q = utils.collision(x, v, s, eta, gamma, L, w)
        v = v - dt * C * Q
        key = jr.PRNGKey(istep)
        train_score_model(model, optimizer, x, v, key, **training_config)

    E = E - jnp.mean(E)
    t += dt
    E_norm = jnp.sqrt(jnp.sum(E ** 2) * eta)
    E_L2.append(E_norm)

#%%
# Phase-space snapshots from x_traj, v_traj
title = fr"Landau damping α={alpha}, k={k}, C={C}, n={n:.0e}, M={M}, Δt={dt}, {score_method}"
outdir_ps = f"data/plots/phase_space/landau_damping_1d_{dv}v/"
fname_ps = f"landau_damping_phase_space_n{n:.0e}_M{M}_dt{dt}_dv{dv}_C{C}_alpha{alpha}_k{k}.png"

fig_ps, path_ps = utils.plot_phase_space_snapshots(
    x_traj, v_traj, t_traj, L, title, outdir_ps, fname_ps
)

plt.show()
plt.close(fig_ps)

# log snapshots
snapshots_dir = os.path.join(path.DATA, "snapshots", f"landau_damping_n{n:.0e}_M{M}_dt{dt}_{score_method}_dv{dv}_C{C}_alpha{alpha}_k{k}")
os.makedirs(snapshots_dir, exist_ok=True)
snapshots_raw_path = os.path.join(snapshots_dir, "snapshots_raw.npz")
np.savez_compressed(
    snapshots_raw_path,
    x_traj=np.array(x_traj, dtype=object),
    v_traj=np.array(v_traj, dtype=object),
    t_traj=np.array(t_traj),
)

# Post-processing: Landau damping fit
t_grid = jnp.linspace(0, final_time, num_steps + 2)

fig_final = plt.figure(figsize=(6, 4))
plt.plot(t_grid, E_L2, marker="o", ms=1, label=f"Simulation (C={C})")

prefactor = -1 / (k ** 3) * jnp.sqrt(jnp.pi / 8) * jnp.exp(-1 / (2 * k ** 2) - 1.5)
pred = jnp.exp(t_grid * prefactor)
pred *= E_L2[0] / pred[0]
plt.plot(t_grid, pred, "k-.", label=fr"collisionless: $e^{{\beta t}}, \beta={float(prefactor):.3f}$")

prefactor_collisional = prefactor - C * jnp.sqrt(2 / (9 * jnp.pi))
predicted_collisional = jnp.exp(t_grid * prefactor_collisional)
predicted_collisional *= E_L2[0] / predicted_collisional[0]
if C > 0:
    plt.plot(
        t_grid,
        predicted_collisional,
        "r--",
        label=fr"collisional: $e^{{\beta t}},\ \beta = {float(prefactor_collisional):.3f}$",
    )

t_np = np.asarray(t_grid)
E_np = np.asarray(E_L2)
mask = (t_np > 0.2) & (t_np < final_time)
t_mask = t_np[mask]
E_mask = E_np[mask]
max_idx = argrelextrema(E_mask, np.greater, order=20)[0]
mt, mv = t_mask[max_idx], E_mask[max_idx]
plt.scatter(mt, mv, c="g", zorder=5)
if len(mt) > 1:
    coeffs = np.polyfit(mt, np.log(mv), 1)
    fit = np.exp(coeffs[1] + coeffs[0] * t_mask)
    plt.plot(t_mask, fit, "g--", label=fr"fitted: $e^{{\beta t}}, \beta={coeffs[0]:.3f}$")

plt.xlabel("Time")
plt.ylabel(r"$||E||_{L^2}$")
plt.title(f"C={C}, n={n:.0e}, Δt={dt}, dv={dv}, α={alpha}, M={M}")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()

outdir = f"data/plots/electric_field_norm/collision_1d_{dv}v/"
os.makedirs(outdir, exist_ok=True)
fname = f"landau_damping_n{n:.0e}_M{M}_dt{dt}_{score_method}_dv{dv}_C{C}_alpha{alpha}_{score_method}.png"
p = os.path.join(outdir, fname)
plt.savefig(p)

plt.show()
plt.close(fig_final)


# %%
