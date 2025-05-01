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
from src.solver import Solver, train_initial_model, psi, evaluate_charge_density

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
    rho = evaluate_charge_density(solver.x, solver.mesh.cells(), solver.eta, solver.mesh.box_lengths[0])
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
alpha = 1  # Perturbation strength
k = 0.5      # Wave number
dx = 1       # Position dimension
dv = 2       # Velocity dimension (as requested)
gamma = -dv
C = 0.
qe = 1.
numerical_constants={"qe": qe, "C": C, "gamma": gamma}

# Create a mesh
box_length = 2 * jnp.pi / k
num_cells = 64 # small number for debugging
mesh = Mesh1D(box_length, num_cells)

# Create initial density distribution
initial_density = CosineNormal(alpha=alpha, k=k, dx=dx, dv=dv)

# Create neural network model
model = MLPScoreModel(dx, dv, hidden_dims=(64, ))

# Number of particles for simulation
num_particles = 1000

# Define training configuration
training_config = {
    "batch_size": 64,
    "num_epochs": 0, # don't train NN when C=0
    "abs_tol": 1e-4,
    "learning_rate": 1e-3,
    "num_batch_steps": 0  # don't train NN when C=0
}

#%%
# eta = mesh.eta

# # 1) Sample particle positions and velocities
# key = jax.random.PRNGKey(seed)
# x, v = initial_density.sample(key, size=num_particles)

# plt.figure()
# xs = jnp.linspace(start=0,stop=box_length,num=100).reshape(-1,1)
# vs = jnp.zeros((100, 2))
# plt.subplot(3, 1, 1)
# vals = initial_density(xs, vs)
# plt.plot(xs, vals / jnp.sum(vals) * box_length, label='density')
# plt.hist(x, density=True, bins=40, label='sample')
# plt.legend()

# # 2) Compute initial charge density
# rho = qe*jax.vmap(lambda cell: jnp.mean(psi(x - cell, eta)))(mesh.cells())

# # 3) Solve Poisson equation
# phi, info = jax.scipy.sparse.linalg.cg(mesh.laplacian(), jnp.mean(rho) - rho)

# # 4) Compute electric field
# E1 = -(jnp.roll(phi, -1) - jnp.roll(phi, 1)) / (2 * mesh.eta[0])
# E = jnp.zeros((E1.shape[0], v.shape[-1]))
# E = E.at[:, 0].set(E1)

# E_int = jnp.cumsum(rho - jnp.mean(rho)) * eta

# plt.subplot(3, 1, 2)
# plt.plot(mesh.cells(), rho - jnp.mean(rho), label='rho')
# plt.plot(mesh.cells(), (jnp.roll(E1, -1) - jnp.roll(E1, 1)) / (2 * eta), label='dE/dx')
# # plt.plot(mesh.cells(), (E_int - jnp.roll(E_int, 1)) / eta, label='dE`/dx')
# plt.legend()

# plt.subplot(3, 1, 3)
# plt.plot(mesh.cells(), E1, label='E')
# plt.plot(mesh.cells(), E_int, label='E`')
# plt.legend()
# plt.show()

# print(f"{x=}")
# print(f"{rho=}")

# what must E satisfy?
# 1. dE/dx = rho - mean(rho)
# 2. E is conservative -- this is always true in 1d

# print((jnp.roll(E1, -1) - jnp.roll(E1, 1)) / (2 * mesh.eta[0]))
# print(rho - jnp.mean(rho))

#%%
solver = Solver(
    mesh=mesh,
    num_particles=num_particles,
    initial_density=initial_density,
    initial_nn=model,
    numerical_constants=numerical_constants,
    seed=seed
)

#%%
def psi(x: jnp.ndarray, eta: jnp.ndarray, L) -> jnp.ndarray:
    "psi_eta(x) = prod_i G(x_i/eta_i) / eta_i, where G(x) = max(0, 1-|x|)."
    x = jnp.min(x, L - x)
    return jnp.prod(jnp.maximum(0., 1. - jnp.abs(x) / eta) / eta, axis=-1)

def evaluate_field_at_particles(x, cells, E, eta):
    """Evaluate electric field at particle positions."""
    return jax.vmap(lambda x_i: eta * jnp.sum(psi(x_i - cells, eta)[:, None] * E, axis=0))(x)

def update_electric_field(E, cells, x, v, eta, dt):
    """Update electric field on the mesh."""
    kernel_values = psi(cells[:, None] - x[None, :], eta)
    return E + dt * jnp.mean(kernel_values[:, :, None] * v, axis=1).reshape(E.shape)

E1 = solver.E[:,0]
E_p = evaluate_field_at_particles(solver.x, mesh.cells(), solver.E, mesh.eta)
x1 = solver.x[:,0]
indices = jnp.argsort(x1)
plt.plot(mesh.cells(), E1, label='E at cells')
plt.plot(x1[indices], E_p[indices, 0], label='E at particles')
plt.legend()
plt.show()

#%%
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

box_length = mesh.box_lengths[0]
rho = qe*jax.vmap(lambda cell: jnp.mean(psi(solver.x - cell, solver.eta, box_length)))(mesh.cells())
plt.plot(mesh.cells(), rho - jnp.mean(rho), label='rho')
E1 = solver.E[:,0]
plt.plot(mesh.cells(), (jnp.roll(E1, -1) - jnp.roll(E1, 1)) / (2 * solver.eta), label='dE/dx')
plt.plot(mesh.cells(), E1, label='E')
plt.legend()
plt.show()

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

for step in range(num_steps):
    # Perform a single time step
    x, v, E = solver.step(x, v, E, dt)
    
    # Calculate metrics
    e_l2_norms[step+1] = jnp.sqrt((jnp.sum(E**2) * solver.eta))[0]
    
    # Print progress
    if (step+1) % 10 == 0:
        print(f"Completed step {step+1}/{num_steps}, L2 norm of E: {e_l2_norms[step+1]:.6f}")
    # if (step+1) % 50 == 0:
plt.plot(times[:step], e_l2_norms[:step])
# plt.yscale('log')
plt.title('L2 Norm of Electric Field')
plt.xlabel('Time')
plt.ylabel('||E||_2')
plt.show()

# Save final state
solver.x, solver.v, solver.E = x, v, E

# Visualize results
visualize_results(solver, mesh, times, e_l2_norms)
# %%
