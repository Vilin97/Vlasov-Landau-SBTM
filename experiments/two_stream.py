#%%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm, trange
from matplotlib import gridspec
import time

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mesh import Mesh1D
from src.density import CosineNormal, TwoStream
from src.score_model import MLPScoreModel
from src.solver import Solver, train_initial_model, psi, evaluate_charge_density, evaluate_field_at_particles, update_positions, update_electric_field
from src.path import ROOT, DATA, PLOTS, MODELS

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
example_name = "two_stream"

#%%
"Initialize constants"
seed = 42

# Set constants
alpha, k, c = 1/200, 1/5, 2.4
dx = 1       # Position dimension
dv = 2       # Velocity dimension
gamma = -dv
C = 0
qe = 1
numerical_constants={"qe": qe, "C": C, "gamma": gamma, "alpha": alpha, "k": k}

# Create a mesh
box_length = 2 * jnp.pi / k
num_cells = 1000
mesh = Mesh1D(box_length, num_cells)

# Number of particles for simulation
num_particles = 10**6

# Create initial density distribution
initial_density = TwoStream(alpha=alpha, k=k, c=c, dx=dx, dv=dv)

# Create neural network model
hidden_dims = (1024,1024)
# model = MLPScoreModel(dx, dv, hidden_dims=hidden_dims)
model = None


# Define training configuration
gd_steps = 40
training_config = {
    "batch_size": 1000,
    "num_epochs": 1000, # initial training
    "abs_tol": 2e-4,
    "learning_rate": 2e-4,
    "num_batch_steps": gd_steps  # at each step
}

cells = mesh.cells()
eta = mesh.eta

#%%
"Initialize the solver"
print(f"C = {C}, N = {num_particles}, num_cells = {num_cells}, box_length = {box_length}, dx = {dx}, dv = {dv}")
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

box_length = mesh.box_lengths[0]
rho = evaluate_charge_density(x0, cells, eta, box_length)
plt.plot(cells, rho - jnp.mean(rho), label='rho')
plt.plot(cells, jnp.gradient(E0, solver.eta[0]), label='dE/dx')
plt.plot(cells, E0, label='E')
plt.legend()
plt.show()

#%%
"Train and save the initial model"
# epochs = solver.training_config["num_epochs"]
# path = os.path.join(MODELS, f'{example_name}_dx{dx}_dv{dv}_alpha{alpha}_k{k}/hidden_{str(hidden_dims)}/epochs_{epochs}')
# if not os.path.exists(path):
#     train_initial_model(model, x0, v0, initial_density,solver.training_config,verbose=True)
#     model.save(path)

# time.sleep(1)  # wait for the model to be saved

# #%%
# "Load the initial model"
# epochs = solver.training_config["num_epochs"]
# path = os.path.join(MODELS, f'{example_name}_dx{dx}_dv{dv}_alpha{alpha}_k{k}/hidden_{str(hidden_dims)}/epochs_{epochs}')
# solver.score_model.load(path)

#%%
"Solve"
# ── simulation parameters ──────────────────────────────────────────────
final_time, dt = 50.0, 0.05
example_path = f"{example_name}_dx{dx}_dv{dv}_C{C}_alpha{alpha}_k{k}_T{final_time}_dt{dt}_N{num_particles}_cells{num_cells}_gd{gd_steps}"
num_steps      = int(final_time / dt)
times          = np.linspace(0.0, final_time, num_steps + 1)

Nx      = mesh.cells().shape[0]
h       = solver.eta[0]                                             # mesh spacing
weights = 1.0                                                       # equal weights

# ── diagnostic storages (only what we will plot) ───────────────────────
EK  = np.empty_like(times)     # kinetic energy
EE  = np.empty_like(times)     # electric energy
S   = np.empty_like(times)     # entropy
E_L2 = np.empty_like(times)    # ‖E‖₂

# phase-space snapshots
snap_idx  = [0, int(10/dt), int(20/dt), int(30/dt), int(40/dt), int(50/dt)]       # indices of times for montage
x_snaps, v1_snaps = [], []

# helper lambdas (JAX) --------------------------------------------------
kin_energy = lambda v: 0.5 * jnp.sum(v.astype(jnp.float32)**2) * (box_length / num_particles)
ele_energy = lambda E: 0.5 * jnp.sum(E.astype(jnp.float32)**2) * h
entropy    = lambda w: -jnp.sum(w * jnp.log(w + 1e-12))
E_L2_func  = lambda E: jnp.sqrt(jnp.sum(E.astype(jnp.float32)**2) * h)

# ── initial diagnostics ────────────────────────────────────────────────
x, v, E = solver.x, solver.v, solver.E
EK[0]   = float(kin_energy(v))
EE[0]   = float(ele_energy(E))
S[0]    = float(entropy(jnp.array([weights])))
E_L2[0] = float(E_L2_func(E))

if 0 in snap_idx:
    x_snaps.append(np.asarray(x).ravel())
    v1_snaps.append(np.asarray(v[:, 0]))

# ── time stepping ───────────────────────────────────────────────────────
print("Running simulation …")
key = jax.random.PRNGKey(seed)
for n in trange(1, num_steps + 1, desc="time-integration"):
    key, sub = jax.random.split(key)
    x, v, E = solver.step(x, v, E, dt, key=sub)

    EK[n]   = float(kin_energy(v))
    EE[n]   = float(ele_energy(E))
    S[n]    = float(entropy(jnp.array([weights])))
    E_L2[n] = float(E_L2_func(E))

    if n in snap_idx:
        x_snaps.append(np.asarray(x).ravel())
        v1_snaps.append(np.asarray(v[:, 0]))

solver.x, solver.v, solver.E = x, v, E  # save final state

#%%
# Save EK, EE, E_L2 to file
statistics_dir = os.path.join(ROOT, "data", "statistics", example_path)
os.makedirs(statistics_dir, exist_ok=True)
np.save(os.path.join(statistics_dir, 'kinetic_energy.npy'), EK)
np.save(os.path.join(statistics_dir, 'electric_energy.npy'), EE)
np.save(os.path.join(statistics_dir, 'electric_field_norm.npy'), E_L2)

#%%
"Plotting"
def plot_electric_field_norm(times, e_l2_norms, example_path, PLOTS):
    plt.figure(figsize=(8, 5))
    plt.plot(times, e_l2_norms, label='||E||_2')

    plt.xlabel('Time')
    plt.ylabel('||E||_2')
    plt.yscale('log')
    plt.title(example_path)
    plt.legend()

    # Ensure the directory exists before saving
    save_dir = os.path.join(PLOTS, "electric_field_norm")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{example_path}.png"), dpi=300)

    plt.show()

def visualize_results(times, EK, EE, S, *, example_name="two_stream"):
    ET = EK + EE
    plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2)
    ax1, ax2, ax3, ax4 = [plt.subplot(g) for g in gs]

    ax1.plot(times, ET); ax1.set_title("Total energy")
    ax2.plot(times, EK); ax2.set_title("Kinetic energy")
    ax3.plot(times, EE); ax3.set_title("Electric energy")
    ax4.plot(times, S ); ax4.set_title("Entropy (WIP)")

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xlabel('t'); ax.grid(alpha=.3)

    plt.tight_layout()
    os.makedirs(PLOTS, exist_ok=True)
    plt.savefig(os.path.join(PLOTS, f"{example_name}_energy_entropy.png"), dpi=300)
    plt.show()
    plt.close()

def plot_phase_space_evolution(xs, vs, L, times,
                               vmax=6, nbins_x=128, nbins_v=128,
                               cmap="jet",
                               example_name="two_stream", num_plot_particles = 100_000):
    """
    xs, vs : list/array with length len(times); each entry shape (N,)
    L      : fundamental period in x
    Only 10^5 particles are used for plotting.
    """
    nt     = len(times)
    fig_h  = nt * 3
    plt.figure(figsize=(6, fig_h))

    for i, (t, x, v) in enumerate(zip(times, xs, vs)):
        N = x.shape[0]
        if N > num_plot_particles:
            idx = np.random.choice(N, num_plot_particles, replace=False)
            x_plot = x[idx]
            v_plot = v[idx]
        else:
            x_plot = x
            v_plot = v

        ax = plt.subplot(nt, 1, i+1)
        H, xe, ve = np.histogram2d(x_plot, v_plot,
                                   bins=[nbins_x, nbins_v],
                                   range=[[0, L], [-vmax, vmax]],
                                   density=True)
        ax.imshow(H.T, origin='lower', aspect='auto',
                  extent=[0, L, -vmax, vmax], cmap=cmap)
        ax.set_ylabel(r"$v_x$")
        ax.set_xlim(0, L); ax.set_ylim(-vmax, vmax)
        ax.set_xticks(np.arange(0, L+0.1, 2*np.pi))
        if i==0:
            ax.set_title(rf"$C={C}$,   t={t}")
        else:
            ax.text(0.02,0.90,rf"$t={t}$", transform=ax.transAxes,
                    ha='left', va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="none"))
    plt.xlabel("x")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"{example_name}_phase_space.png"), dpi=300)
    plt.show()
    plt.close()

#%%
plot_electric_field_norm(times, E_L2, example_path, PLOTS)
visualize_results(times, EK, EE, S)
#%%
plot_phase_space_evolution(x_snaps, v1_snaps,
                           L=mesh.box_lengths[0],
                           times=times[list(snap_idx)])
# %%
