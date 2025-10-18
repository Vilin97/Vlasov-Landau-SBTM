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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_enable_x64", True)

example_name = "landau_damping"

def visualize_results(solver, mesh, times, e_l2_norms, n_scatter=100_000, save=True):
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
    
    if save:
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
C = 0.0      # Collision strength 
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
gd_steps = 5
batch_size = 2**10
num_batch_steps = gd_steps
training_config = {
    "batch_size": batch_size,
    "num_epochs": 1000, # initial training
    "abs_tol": 1e-3,
    "learning_rate": 1e-4,
    "num_batch_steps": gd_steps  # at each step
}

cells = mesh.cells()
eta = mesh.eta

#%%
"Initialize the solver"
print(f"C = {C}, alpha={alpha}, N = {num_particles}, num_cells = {num_cells}, box_length = {box_length}, dx = {dx}, dv = {dv}")
print(f"Learning rate = {training_config['learning_rate']}, batch size = {training_config['batch_size']}, num_batch_steps = {training_config['num_batch_steps']}")
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

time.sleep(1)
#%%
solver.score_model.load(path)

# Plot the true score and the score given by the model for a subset of particles
num_plot = 100_000
v_plot = v0[:num_plot]
x_plot = x0[:num_plot]
sort_idx = np.argsort(v_plot[:, 0])

true_score = initial_density.score(x, v)
model_score = solver.score_model(x, v)

plt.figure(figsize=(8, 2))
plt.plot(v_plot[sort_idx, 0], true_score[sort_idx, 0], label=f'True Score', alpha=0.7)
plt.plot(v_plot[sort_idx, 0], model_score[sort_idx, 0], label=f'Model Score, mse={loss.mse(model_score, true_score):.4f}', alpha=0.7)
plt.xlabel('Velocity $v_1$')
plt.ylabel('Score')
plt.title('True Score vs Model Score')
plt.legend()
plt.show()

#%%
"Solve"
# Simulation parameters
key = jax.random.PRNGKey(seed)
final_time = 20.0
dt = 0.01
num_steps = int(final_time / dt)

example_path = f"{example_name}_dx{dx}_dv{dv}_C{C}_alpha{alpha}_k{k}_T{final_time}_dt{dt}_N{num_particles}_cells{num_cells}_gd{gd_steps}"

# Arrays to store metrics over time
times = np.linspace(0, final_time, num_steps+1)
e_l2_norms = np.zeros(num_steps+1)
implicit_losses = np.zeros(num_steps+1)
train_losses = []

# Calculate initial metrics
e_l2_norms[0] = jnp.sqrt((jnp.sum(solver.E**2) * solver.eta))[0]
implicit_losses[0] = loss.implicit_score_matching_loss(solver.score_model, x[:batch_size], v[:batch_size], key=jax.random.PRNGKey(seed))

collision_strengths = np.zeros(num_steps+1)
collision_strengths[0] = jnp.linalg.norm(collision(solver.x, solver.v, solver.score_model(x, v), solver.eta, C, gamma, box_length, num_cells))
drift_strengths = np.zeros(num_steps+1)
drift_strengths[0] = jnp.linalg.norm(evaluate_field_at_particles(x, cells, solver.E, eta, box_length))

for step in tqdm(range(num_steps), desc="Solving"):
    # Perform a single time step
    key, subkey = jax.random.split(key)
    # x, v, E = solver.step(x, v, E, dt, key=subkey)

    t1 = time.time()
    E_at_particles = evaluate_field_at_particles(x, cells, E, eta, box_length)
    
    t2 = time.time()
    v_new = v.at[:, 0].add(dt * E_at_particles)
    t3 = time.time()
    if C > 0:
        losses = train_score_model(solver.score_model, solver.optimizer, solver.loss_fn, x, v, key, batch_size, num_batch_steps)
    else:
        losses = []
    t4 = time.time()
    if C > 0:
        s = solver.score_model(x, v)
    else:
        s = 0
    t5 = time.time()
    if C > 0:
        collision_term = collision(x, v, s, eta, C, gamma, box_length, num_cells)
    else:
        collision_term = 0.0
    v_new = v_new - dt * collision_term

    t6 = time.time()
    x_new = update_positions(x, v_new, dt, box_length)
    E_new = update_electric_field(E, cells, x_new, v_new, eta, dt, box_length)

    x = jnp.mod(x_new, box_length)  # Ensure periodic boundary conditions
    v = v_new
    E = E_new
    
    t7 = time.time()
    # Calculate metrics
    e_l2_norms[step+1] = jnp.sqrt((jnp.sum(E**2) * solver.eta))[0]
    collision_strengths[step+1] = jnp.linalg.norm(collision_term)
    drift_strengths[step+1] = jnp.linalg.norm(E_at_particles)
    t8 = time.time()
    implicit_losses[step+1] = loss.implicit_score_matching_loss(solver.score_model, x[:batch_size], v[:batch_size], key=jax.random.PRNGKey(seed))
    train_losses.append(losses)
    t9 = time.time()

    # Print timing information
    if step % 200 == 0:
        print(f"Step {step+1}/{num_steps}:")
        print(f"  E at particles: {t2 - t1:.6f}s")
        print(f"  Update velocities: {t3 - t2:.6f}s")
        print(f"  Train score model: {t4 - t3:.6f}s")
        print(f"  Score evaluation: {t5 - t4:.6f}s")
        print(f"  Collision term: {t6 - t5:.6f}s")
        print(f"  Update positions and field: {t7 - t6:.6f}s")
        print(f"  Calculate metrics: {t8 - t7:.6f}s")
        print(f"  Calculate implicit loss: {t9 - t8:.6f}s")
        print(f"  Total time for step: {t9 - t1:.6f}s")
        print(f"Bottleneck: {max([(t2 - t1, 'E at particles'), (t3 - t2, 'Update velocities'), (t4 - t3, 'Train score model'), (t5 - t4, 'Score evaluation'), (t6 - t5, 'Collision term'), (t7 - t6, 'Update positions and field'), (t8 - t7, 'Calculate metrics'), (t9 - t8, 'Calculate implicit loss')], key=lambda x: x[0])}")
    
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

    # Predicted collisional curve
    t_grid = jnp.linspace(0, times[-1], len(times))
    k = solver.numerical_constants["k"]
    C = solver.numerical_constants["C"]
    prefactor = -1/(k**3) * jnp.sqrt(jnp.pi/8) * jnp.exp(-1/(2*k**2) - 1.5)
    prefactor_collisional = prefactor - C * jnp.sqrt(2/(9*jnp.pi))
    predicted_collisional = jnp.exp(t_grid * prefactor_collisional)
    predicted_collisional *= loaded_e_l2_norms[0] / predicted_collisional[0]
    gamma = prefactor_collisional
    if C > 0:
        plt.plot(t_grid, predicted_collisional, 'r--', label=fr'$collisional: e^{{\gamma t}},\ \gamma = {gamma:.3f}$')

    # Predicted collisionless curve (C=0)
    prefactor_collisionless = prefactor  # C=0
    predicted_collisionless = jnp.exp(t_grid * prefactor_collisionless)
    predicted_collisionless *= loaded_e_l2_norms[0] / predicted_collisionless[0]
    gamma0 = prefactor_collisionless
    plt.plot(t_grid, predicted_collisionless, 'b--', label=fr'$collisionless: e^{{\gamma_0 t}},\ \gamma_0 = {gamma0:.3f}$')

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
    mask = times < 20
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
plt.plot(times, collision_strengths, label='Collision Strength')
plt.plot(times, drift_strengths, label='Drift Strength')
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Strength')
plt.legend()
plt.show()

# Plot training losses
train_losses_flat = [loss for losses in train_losses for loss in losses]
refined_times = np.linspace(0, final_time, len(train_losses_flat))
plt.plot(refined_times, train_losses_flat, label=f'Training Loss, batch size {batch_size}', alpha=0.7)
plt.plot(times, implicit_losses, label=f'Implicit Loss, batch size {batch_size}', alpha=1)
plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Implicit Loss and Training Loss Over Time')
plt.legend()
plt.show()
#%%
# Visualize results
visualize_results(solver, mesh, times, e_l2_norms, save=False)

# %%
