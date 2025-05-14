#%%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

from src.mesh import Mesh1D
from src.density import CosineNormal
from src.score_model import MLPScoreModel
from src.solver import Solver, train_initial_model, psi, evaluate_charge_density, evaluate_field_at_particles, update_positions, update_electric_field

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def visualize_results(solver, mesh, times, e_l2_norms):
    """Visualize the results of the solver simulation."""
    dt = times[1] - times[0]
    num_particles = solver.x.shape[0]
    dx = solver.x.shape[1]
    dv = solver.v.shape[1]
    num_cells = mesh.cells().shape[0]
    alpha, k = solver.numerical_constants["alpha"], solver.numerical_constants["k"]
    
    plt.figure(figsize=(18, 10))
    
    # Plot 1: Final particle phase space (position and first velocity component)
    plt.subplot(2, 3, 1)
    plt.scatter(solver.x.flatten(), solver.v[:, 0], s=1, alpha=0.3)
    plt.title('Final Particle Phase Space (x, v1)')
    plt.xlabel('Position (x)')
    plt.ylabel('Velocity (v1)')
    
    # Plot 2: Final particle distribution in v1-v2 space
    plt.subplot(2, 3, 2)
    plt.scatter(solver.v[:, 0], solver.v[:, 1], s=1, alpha=0.3)
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
    plt.hist(solver.v[:, 0], bins=30, alpha=0.7)
    plt.title('Velocity Distribution (v1)')
    plt.xlabel('Velocity')
    plt.ylabel('Count')
    
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save figure to plots directory
    filename = f'landau_damping_dx{dx}_dv{dv}_alpha{alpha}_k{k}_dt{dt}_N{num_particles}_cells{num_cells}.png'
    plt.savefig(os.path.join(plots_dir, filename))
    plt.show()

#%%
# Set random seed for reproducibility
seed = 42

# Set constants
alpha = 0.1  # Perturbation strength
k = 0.5      # Wave number
dx = 1       # Position dimension
dv = 3       # Velocity dimension
gamma = -dv
C = 1.0
qe = 1.
numerical_constants={"qe": qe, "C": C, "gamma": gamma, "alpha": alpha, "k": k}

# Create a mesh
box_length = 2 * jnp.pi / k
num_cells = 256
mesh = Mesh1D(box_length, num_cells)

# Create initial density distribution
initial_density = CosineNormal(alpha=alpha, k=k, dx=dx, dv=dv)

# Create neural network model
model = MLPScoreModel(dx, dv, hidden_dims=(64, ))

# Number of particles for simulation
num_particles = 100_000

# Define training configuration
training_config = {
    "batch_size": 1000,
    "num_epochs": 10, # initial training
    "abs_tol": 1e-3,
    "learning_rate": 1e-3,
    "num_batch_steps": 10  # at each step
}

cells = mesh.cells()
eta = mesh.eta

#%%
print(f"N = {num_particles}, num_cells = {num_cells}, box_length = {box_length}, dx = {dx}, dv = {dv}")
solver = Solver(
    mesh=mesh,
    num_particles=num_particles,
    initial_density=initial_density,
    initial_nn=model,
    numerical_constants=numerical_constants,
    seed=seed
)
solver.training_config = training_config
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
path = os.path.expanduser(f'~/Vlasov-Landau-SBTM/data/score_models/landau_damping_dx{dx}_dv{dv}_alpha{alpha}_k{k}')
if not os.path.exists(path):
    train_initial_model(model, x0, v0, initial_density,training_config,verbose=True)
    model.save(path)

#%%
path = os.path.expanduser(f'~/Vlasov-Landau-SBTM/data/score_models/landau_damping_dx{dx}_dv{dv}_alpha{alpha}_k{k}')
solver.score_model.load(path)

#%%
"Solve"
# Simulation parameters
final_time = 10.0 # set to 10 later
dt = 0.02
num_steps = int(final_time / dt)

# Arrays to store metrics over time
times = np.linspace(0, final_time, num_steps+1)
e_l2_norms = np.zeros(num_steps+1)

# Calculate initial metrics
e_l2_norms[0] = jnp.sqrt((jnp.sum(solver.E**2) * solver.eta))[0]

# Run simulation for multiple steps and collect metrics
print("Running simulation...")
x, v, E = solver.x, solver.v, solver.E

for step in tqdm(range(num_steps), desc="Solving"):
    # Perform a single time step
    
    x, v, E = solver.step(x, v, E, dt)
    
    # Calculate metrics
    e_l2_norms[step+1] = jnp.sqrt((jnp.sum(E**2) * solver.eta))[0]
    
    # Print progress
    # if step % 20 == 0:
    #     print(f"Completed step {step+1}/{num_steps}, L2 norm of E: {e_l2_norms[step+1]:.6f}")
    #     plt.plot(mesh.cells(), E, label='E')
    #     plt.legend()
    #     plt.show()

# Save final state
solver.x, solver.v, solver.E = x, v, E

#%%
# Visualize results
visualize_results(solver, mesh, times, e_l2_norms)


# %%
# NOTE: `collision` takes time O(num_particles/num_cells). So increasing num_cells will decrease the time taken!

import time
from src.solver import collision

s = model(x0, v0)
collision(x0, v0, s, eta, 1., gamma, box_length, num_cells)

t0 = time.time()
s = model(x0, v0).block_until_ready()
t1 = time.time()
collision(x0, v0, s, eta, 1., gamma, box_length, num_cells).block_until_ready()
t2 = time.time()

print(f"Model time: {t1 - t0:.4f} seconds")
print(f"Collision time: {t2 - t1:.4f} seconds")
