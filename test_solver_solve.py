#%%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx
import os
import numpy as np
from tqdm import tqdm

from src.mesh import Mesh1D
from src.density import CosineNormal
from src.score_model import MLPScoreModel
from src.solver import Solver, train_initial_model, psi

def calculate_charge_density(solver):
    """Calculate the charge density on the mesh."""
    qe = solver.numerical_constants["qe"]
    return qe * jax.vmap(lambda cell: jnp.mean(
        psi(solver.x - cell, solver.eta)))(solver.mesh.cells())

def visualize_results(solver, mesh, times, e_l2_norms):
    """Visualize the results of the solver simulation."""
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
    rho = calculate_charge_density(solver)
    plt.plot(x_cells, rho)
    plt.title('Final Charge Density')
    plt.xlabel('Position (x)')
    plt.ylabel('Density (ρ)')
    
    # Plot 4: Final electric field
    plt.subplot(2, 3, 4)
    plt.plot(x_cells, solver.E)
    plt.title('Final Electric Field')
    plt.xlabel('Position (x)')
    plt.ylabel('Electric Field (E)')
    
    # Plot 5: L2 norm of E over time
    plt.subplot(2, 3, 5)
    plt.plot(times, e_l2_norms)
    plt.title('L2 Norm of Electric Field')
    plt.xlabel('Time')
    plt.ylabel('||E||_2')
    
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
    plt.savefig(os.path.join(plots_dir, 'solver_solve.png'))
    plt.show()

#%%
# Set random seed for reproducibility
seed = 42

# Set constants
alpha = 0.1  # Perturbation strength
k = 0.5      # Wave number
dx = 1       # Position dimension
dv = 2       # Velocity dimension (as requested)
gamma = -dv
C = 0.
qe = 1.
numerical_constants={"qe": qe, "C": C, "gamma": gamma}

# Create a mesh
box_length = 2 * jnp.pi / k
num_cells = 128
mesh = Mesh1D(box_length, num_cells)

# Create initial density distribution
initial_density = CosineNormal(alpha=alpha, k=k, dx=dx, dv=dv)

# Create neural network model
model = MLPScoreModel(dx, dv, hidden_dims=(64, ))

# Number of particles for simulation
num_particles = 100000

# Define training configuration
training_config = {
    "batch_size": 64,
    "num_epochs": 0, # don't train NN when C=0
    "abs_tol": 1e-4,
    "learning_rate": 1e-3,
    "num_batch_steps": 0  # don't train NN when C=0
}

# Initialize the solver
print("Initializing solver...")
solver = Solver(
    mesh=mesh,
    num_particles=num_particles,
    initial_density=initial_density,
    initial_nn=model,
    numerical_constants=numerical_constants,
    seed=seed
)

# Add training_config to solver (needed for the step method)
solver.training_config = training_config

# Train the initial model
print("Training initial model...")
train_initial_model(model, solver.x, solver.v, initial_density, training_config)

# Simulation parameters
final_time = 10.0
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

#%%
for step in tqdm(range(num_steps)):
    # Perform a single time step
    x, v, E = solver.step(x, v, E, dt)
    
    # Calculate metrics
    e_l2_norms[step+1] = jnp.sqrt((jnp.sum(E**2) * solver.eta))[0]
    
    # Print progress
    if (step+1) % 10 == 0:
        print(f"Completed step {step+1}/{num_steps}, L2 norm of E: {e_l2_norms[step+1]:.6f}")
    if (step+1) % 50 == 0:
        plt.plot(times[:step], e_l2_norms[:step])
        plt.title('L2 Norm of Electric Field')
        plt.xlabel('Time')
        plt.ylabel('||E||_2')
        plt.show()

# Save final state
solver.x, solver.v, solver.E = x, v, E

# Visualize results
visualize_results(solver, mesh, times, e_l2_norms)