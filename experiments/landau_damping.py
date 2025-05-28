#%%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mesh import Mesh1D
from src.density import CosineNormal
from src.score_model import MLPScoreModel
from src.solver import Solver, train_initial_model, psi, evaluate_charge_density, evaluate_field_at_particles, update_positions, update_electric_field
from src.path import ROOT, DATA, PLOTS, MODELS
from scipy.signal import argrelextrema

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

example_name = "landau_damping"

def visualize_results(solver, mesh, times, e_l2_norms, n_scatter=100_000):
    """Visualize the results of the solver simulation."""
    
    plt.figure(figsize=(18, 10))
    idx = np.random.choice(solver.x.shape[0], n_scatter, replace=False)
    v = solver.v[idx]
    
    # Plot 1: Final particle phase space (position and first velocity component)
    plt.subplot(2, 3, 1)
    plt.scatter(solver.x[idx].flatten(), solver.v[idx, 0], s=1, alpha=0.3)
    plt.title('Final Particle Phase Space (x, v1)')
    plt.xlabel('Position (x)')
    plt.ylabel('Velocity (v1)')
    
    # Plot 2: Final particle distribution in v1-v2 space
    plt.subplot(2, 3, 2)
    plt.scatter(v[:, 0], v[:, 1], s=1, alpha=0.3)
    plt.title('Final Velocity Distribution')
    plt.xlabel('Velocity (v1)')
    plt.ylabel('Velocity (v2)')
    
    # Plot 3: Final charge density
    plt.subplot(2, 3, 3)
    x_cells = mesh.cells().flatten()
    rho = evaluate_charge_density(solver.x, solver.mesh.cells(), solver.eta, solver.mesh.box_lengths[0])
    plt.plot(x_cells, rho)
    plt.title('Final Charge Density')
    plt.xlabel('Position (x)')
    plt.ylabel('Density (ρ)')
    
    # Plot 4: Final electric field
    plt.subplot(2, 3, 4)
    plt.plot(x_cells, solver.E, label="E")
    plt.title('Final Electric Field')
    plt.xlabel('Position (x)')
    plt.ylabel('Electric Field (E)')
    plt.legend()
    
    # Plot 5: L2 norm of E over time
    plt.subplot(2, 3, 5)
    plt.plot(times, e_l2_norms, label='Simulated')
    # Predicted curve
    t_grid = jnp.linspace(0, times[-1], len(times))
    prefactor = -1/(k**3) * jnp.sqrt(jnp.pi/8) * jnp.exp(-1/(2*k**2) - 1.5)
    prefactor -= solver.numerical_constants["C"] * jnp.sqrt(2/(9*jnp.pi))
    predicted = jnp.exp(t_grid * prefactor)
    predicted *= e_l2_norms[0] / predicted[0]
    gamma = prefactor
    plt.plot(t_grid, predicted, 'r--', label=fr'$e^{{\gamma t}},\ \gamma = {gamma:.3f}$')
    plt.title('L2 Norm of Electric Field')
    plt.xlabel('Time')
    plt.ylabel('||E||_2')
    plt.yscale('log')
    plt.legend()
    
    # Plot 6: Histogram of velocities (v1 component)
    plt.subplot(2, 3, 6)
    plt.hist(v[:, 0], bins=30, alpha=0.7)
    plt.title('Velocity Distribution (v1)')
    plt.xlabel('Velocity')
    plt.ylabel('Count')
    
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    plots_dir = PLOTS
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save figure to plots directory
    filename = f'{example_path}.png'
    plt.savefig(os.path.join(plots_dir, filename))
    plt.show()

#%%
"Initialize parameters"
seed = 42

# Set constants
alpha = 0.1  # Perturbation strength
k = 0.5      # Wave number
dx = 1       # Position dimension
dv = 2       # Velocity dimension
gamma = -dv
C = 0.05
qe = 1
numerical_constants={"qe": qe, "C": C, "gamma": gamma, "alpha": alpha, "k": k}

# Create a mesh
box_length = 2 * jnp.pi / k
num_cells = 1000
mesh = Mesh1D(box_length, num_cells)

# Number of particles for simulation
num_particles = 10**6 # 10^8 takes ~16Gb of VRAM, collisionless

# Create initial density distribution
initial_density = CosineNormal(alpha=alpha, k=k, dx=dx, dv=dv)

# Create neural network model
hidden_dims = (1024,)
model = MLPScoreModel(dx, dv, hidden_dims=hidden_dims)
# model = None


# Define training configuration
gd_steps = 10
training_config = {
    "batch_size": 1000,
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

box_length = mesh.box_lengths[0]
rho = evaluate_charge_density(x0, cells, eta, box_length)
plt.plot(cells, rho - jnp.mean(rho), label='rho')
plt.plot(cells, jnp.gradient(E0, solver.eta[0]), label='dE/dx')
plt.plot(cells, E0, label='E')
plt.legend()
plt.show()

#%%
# Train and save the initial model
epochs = solver.training_config["num_epochs"]
path = os.path.join(MODELS, f'landau_damping_dx{dx}_dv{dv}_alpha{alpha}_k{k}/hidden_{str(hidden_dims)}/epochs_{epochs}')
if not os.path.exists(path):
    train_initial_model(model, x0, v0, initial_density, solver.training_config, verbose=True)
    model.save(path)

#%%
solver.score_model.load(path)

#%%
"Solve"
# Simulation parameters
final_time = 60.0
dt = 0.01
num_steps = int(final_time / dt)

example_path = f"{example_name}_dx{dx}_dv{dv}_C{C}_alpha{alpha}_k{k}_T{final_time}_dt{dt}_N{num_particles}_cells{num_cells}_gd{gd_steps}"

# Arrays to store metrics over time
times = np.linspace(0, final_time, num_steps+1)
e_l2_norms = np.zeros(num_steps+1)

# Calculate initial metrics
e_l2_norms[0] = jnp.sqrt((jnp.sum(solver.E**2) * solver.eta))[0]

# Run simulation for multiple steps and collect metrics
print("Running simulation...")
x, v, E = solver.x, solver.v, solver.E

key = jax.random.PRNGKey(seed)
for step in tqdm(range(num_steps), desc="Solving"):
    # Perform a single time step
    key, subkey = jax.random.split(key)
    x, v, E = solver.step(x, v, E, dt, key=subkey)
    
    # Calculate metrics
    e_l2_norms[step+1] = jnp.sqrt((jnp.sum(E**2) * solver.eta))[0]
    
# Save final state
solver.x, solver.v, solver.E = x, v, E

# Save e_l2_norms to file
statistics_dir = os.path.join(ROOT, "data", "statistics", example_path)
os.makedirs(statistics_dir, exist_ok=True)
stats_filename = f'electric_field_norm.npy'
np.save(os.path.join(statistics_dir, stats_filename), e_l2_norms)

# %%
def plot_electric_field_norm(times, loaded_e_l2_norms, example_path, solver, PLOTS):
    """
    Load and plot the electric field norm over time with predicted curve and maxima fit.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(times, loaded_e_l2_norms, label='||E||_2')

    # Predicted curve
    t_grid = jnp.linspace(0, times[-1], len(times))
    k = solver.numerical_constants["k"]
    C = solver.numerical_constants["C"]
    prefactor = -1/(k**3) * jnp.sqrt(jnp.pi/8) * jnp.exp(-1/(2*k**2) - 1.5)
    prefactor -= C * jnp.sqrt(2/(9*jnp.pi))
    predicted = jnp.exp(t_grid * prefactor)
    predicted *= loaded_e_l2_norms[0] / predicted[0]
    gamma = prefactor
    plt.plot(t_grid, predicted, 'r--', label=fr'$e^{{\gamma t}},\ \gamma = {gamma:.3f}$')

    plt.xlabel('Time')
    plt.ylabel('||E||_2')
    plt.yscale('log')
    plt.title(example_path)
    plt.legend()

    # Set y-limits to min/max of the electric field norm plus a bit, ensuring positivity for log scale
    ymin = loaded_e_l2_norms[loaded_e_l2_norms > 0].min()
    ymax = loaded_e_l2_norms.max()
    plt.ylim(ymin * 0.95, ymax * 1.05)

    # Find all local maxima for t < ... and plot a straight line through them
    mask = times < 10
    norms_masked = loaded_e_l2_norms[mask]
    times_masked = times[mask]

    # Find local maxima indices
    maxima_indices = argrelextrema(norms_masked, np.greater, order=1)[0]
    maxima_times = times_masked[maxima_indices]
    maxima_values = norms_masked[maxima_indices]

    # Plot maxima points
    plt.scatter(maxima_times, maxima_values, color='g', marker='o')

    # Fit a straight line (in log space) through the maxima
    if len(maxima_times) > 1:
        coeffs = np.polyfit(maxima_times, np.log(maxima_values), 1)
        fit_line = np.exp(coeffs[1] + coeffs[0] * times_masked)
        plt.plot(times_masked, fit_line, 'g--', label=fr'$e^{{\beta t}}, \beta={coeffs[0]:.3f}$')
        print(f"Decay rate from maxima fit: {coeffs[0]:.4f}")

    plt.legend()

    # Ensure the directory exists before saving
    save_dir = os.path.join(PLOTS, "electric_field_norm")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{example_path}.png"), dpi=300)

    plt.show()

#%%
plot_electric_field_norm(times, e_l2_norms, example_path, solver, PLOTS)
#%%
# Visualize results
visualize_results(solver, mesh, times, e_l2_norms)