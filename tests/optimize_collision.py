#%%
import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import src.path
from src.mesh import Mesh1D
from src.density import CosineNormal
from src.score_model import MLPScoreModel, kde_score_hat
from src.solver import Solver, train_initial_model, psi, evaluate_charge_density, evaluate_field_at_particles, update_positions, update_electric_field, collision, train_score_model
import src.loss as loss
from src.path import ROOT, DATA, PLOTS, MODELS
from scipy.signal import argrelextrema
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#%%
"Initialize parameters"
seed = 42

# Set constants
alpha = 0.1  # Perturbation strength
k = 0.5      # Wave number
dx = 1       # Position dimension
dv = 2       # Velocity dimension
gamma = -dv
C = 0.1
qe = 1
numerical_constants={"qe": qe, "C": C, "gamma": gamma, "alpha": alpha, "k": k}

# Create a mesh
box_length = 2 * jnp.pi / k
num_cells = 512
mesh = Mesh1D(box_length, num_cells)

# Number of particles for simulation
num_particles = 2**20 # 10^8 takes ~16Gb of VRAM, collisionless

# Create initial density distribution
initial_density = CosineNormal(alpha=alpha, k=k, dx=dx, dv=dv)

# Create neural network model
hidden_dims = (1024,)
model = MLPScoreModel(dx, dv, hidden_dims=hidden_dims)
# model = None


# Define training configuration
gd_steps = 0
batch_size = 2**12
num_batch_steps = gd_steps
training_config = {
    "batch_size": 2**12,
    "num_epochs": 1000, # initial training
    "abs_tol": 1e-4,
    "learning_rate": 1e-4,
    "num_batch_steps": gd_steps  # at each step
}

cells = mesh.cells()
eta = mesh.eta

#%%
"Initialize the solver"
print(f"C = {C}, alpha={alpha}, N = {num_particles}, num_cells = {num_cells}, box_length = {box_length}, dx = {dx}, dv = {dv}")
solver = Solver(
    mesh=mesh,
    num_particles=num_particles,
    initial_density=initial_density,
    initial_nn=model,
    numerical_constants=numerical_constants,
    seed=seed,
    training_config=training_config
)
x0, v0, E0 = solver.x, solver.v, solver.E
x, v, E = x0, v0, E0

box_length = mesh.box_lengths[0]
rho = evaluate_charge_density(x0, cells, eta, box_length)

#%%
# Train and save the initial model
epochs = solver.training_config["num_epochs"]
path = os.path.join(MODELS, f'landau_damping_dx{dx}_dv{dv}_alpha{alpha}_k{k}/hidden_{str(hidden_dims)}/epochs_{epochs}')
if not os.path.exists(path):
    train_initial_model(model, x0, v0, initial_density, solver.training_config, verbose=True)
    model.save(path)

time.sleep(1)
solver.score_model.load(path)
s = solver.score_model(x, v)

#%%
from functools import partial
import jax, jax.numpy as jnp
from jax import lax
from jax import vmap

@jax.jit
def psi(x, eta, box_length):
    x = (x + 0.5 * box_length) % box_length - 0.5 * box_length   # centered_mod
    return jnp.maximum(0.0, 1.0 - jnp.abs(x / eta)) / eta

@jax.jit
def A(z, C, gamma):
    z_norm  = jnp.linalg.norm(z) + 1e-10
    factor  = C * z_norm ** gamma
    return factor * (jnp.eye(z.shape[-1]) * z_norm**2 - jnp.outer(z, z))

@partial(jax.jit, static_argnames='num_cells')
def collision(x, v, s, eta, C, gamma, box_length, num_cells):
    """
    Q_i = (L/N) Σ_{|x_i−x_j|≤η} ψ(x_i−x_j) · A(v_i−v_j)(s_i−s_j)
          with the linear-hat kernel ψ of width η, periodic on [0,L].

    Complexity O(N η/L)  (exact for the hat kernel).
    """
    x = x[:, 0]
    N, d      = v.shape
    M         = num_cells
    w_particle = box_length / N        # (= L/N)
    # w_particle = 1 / N        # (= L/N)

    # ---- bin & sort particles by cell --------------------------------------
    idx    = jnp.floor(x / eta).astype(jnp.int32) % M
    order  = jnp.argsort(idx)
    x_s, v_s, s_s, idx_s = x[order], v[order], s[order], idx[order]

    counts   = jnp.bincount(idx_s, length=M)
    cell_ofs = jnp.cumsum(jnp.concatenate([jnp.array([0]), counts[:-1]]))

    # ---- per-particle collision using lax.fori_loop (no dynamic slicing) ---
    def Q_single(i):
        xi, vi, si = x_s[i], v_s[i], s_s[i]
        cell       = idx_s[i]
        Q_i        = jnp.zeros(d)

        def loop_over_cell(c, acc):
            start = cell_ofs[c]
            end   = start + counts[c]

            def loop_over_particles(j, inner_acc):
                xj, vj, sj = x_s[j], v_s[j], s_s[j]
                ψ  = psi(xi - xj, eta, box_length)
                dv = vi - vj
                ds = si - sj
                inner_acc += ψ * jnp.dot(A(dv, C, gamma), ds)
                return inner_acc

            acc = lax.fori_loop(start, end, loop_over_particles, acc)
            return acc

        # neighbour cells that overlap the hat (periodic)
        for c in ((cell - 1) % M, cell, (cell + 1) % M):
            Q_i = loop_over_cell(c, Q_i)

        return Q_i

    Q_sorted = jax.vmap(Q_single)(jnp.arange(N))

    # ---- unsort back to original particle order ----------------------------
    rev = jnp.empty_like(order)
    rev = rev.at[order].set(jnp.arange(N))
    return w_particle * Q_sorted[rev]

#%%
def A_apply(dv, ds, C, gamma, eps=1e-10):
    r2   = jnp.dot(dv, dv) + eps          # ‖dv‖²
    r_g  = r2 ** (gamma / 2)              # ‖dv‖^γ
    dvds = jnp.dot(dv, ds)                # dv·ds
    return C * r_g * (r2 * ds - dvds * dv)

@partial(jax.jit, static_argnames='num_cells')
def collision_2(x, v, s, eta, C, gamma, box_length, num_cells):
    """
    Q_i = (L/N) Σ_{|x_i−x_j|≤η} ψ(x_i−x_j) · A(v_i−v_j)(s_i−s_j)
          with the linear-hat kernel ψ of width η, periodic on [0,L].

    Complexity O(N η/L)  (exact for the hat kernel).
    """
    x = x[:, 0]
    N, d      = v.shape
    M         = num_cells
    w_particle = box_length / N        # (= L/N)
    # w_particle = 1 / N        # (= L/N)

    # ---- bin & sort particles by cell --------------------------------------
    idx    = jnp.floor(x / eta).astype(jnp.int32) % M
    order  = jnp.argsort(idx)
    x_s, v_s, s_s, idx_s = x[order], v[order], s[order], idx[order]

    counts   = jnp.bincount(idx_s, length=M)
    cell_ofs = jnp.cumsum(jnp.concatenate([jnp.array([0]), counts[:-1]]))

    # ---- per-particle collision using lax.fori_loop (no dynamic slicing) ---
    def Q_single(i):
        xi, vi, si = x_s[i], v_s[i], s_s[i]
        cell       = idx_s[i]
        Q_i        = jnp.zeros(d)

        def loop_over_cell(c, acc):
            start = cell_ofs[c]
            end   = start + counts[c]

            def loop_over_particles(j, inner_acc):
                xj, vj, sj = x_s[j], v_s[j], s_s[j]
                ψ  = psi(xi - xj, eta, box_length)
                dv = vi - vj
                ds = si - sj
                inner_acc += ψ * A_apply(dv, ds, C, gamma)
                return inner_acc

            acc = lax.fori_loop(start, end, loop_over_particles, acc)
            return acc

        # neighbour cells that overlap the hat (periodic)
        for c in ((cell - 1) % M, cell, (cell + 1) % M):
            Q_i = loop_over_cell(c, Q_i)

        return Q_i

    Q_sorted = jax.vmap(Q_single)(jnp.arange(N))

    # ---- unsort back to original particle order ----------------------------
    rev = jnp.empty_like(order)
    rev = rev.at[order].set(jnp.arange(N))
    return w_particle * Q_sorted[rev]

#%%
collision_3(x, v, s, eta, C, gamma, box_length, num_cells, 2500)

# Compute collision results from both functions
t1 = time.time()
result_collision = collision_2(x, v, s, eta, C, gamma, box_length, num_cells).block_until_ready()
t2 = time.time()
print(f"Original collision took {t2 - t1:.4f} seconds")

t1 = time.time()
result_collision_new = collision_3(x, v, s, eta, C, gamma, box_length, num_cells, 2500).block_until_ready()
t2 = time.time()
print(f"new collision took {t2 - t1:.4f} seconds")

# Compute relative error
abs_diff = jnp.abs(result_collision - result_collision_new)
norm_collision = jnp.linalg.norm(result_collision)
norm_diff = jnp.linalg.norm(abs_diff)
relative_error = norm_diff / norm_collision

print(f"Relative error (L2 norm): {relative_error * 100:.6f}%")
print(f"Max absolute difference: {jnp.max(abs_diff):.6e}")
print(f"Mean absolute difference: {jnp.mean(abs_diff):.6e}")
# %%
s = solver.score_model(x, v)
collision(x, v, s, eta, C, gamma, box_length, num_cells).block_until_ready()

with jax.profiler.trace(DATA+"/tmp/jax-trace", create_perfetto_link=False):
    _ = collision(x, v, s, eta, C, gamma, box_length, num_cells).block_until_ready()