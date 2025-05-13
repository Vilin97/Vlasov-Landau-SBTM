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
    plt.plot(x_cells, solver.E, label=["E1", "E2"])
    plt.title('Final Electric Field')
    plt.xlabel('Position (x)')
    plt.ylabel('Electric Field (E)')
    plt.legend()
    
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
dv = 1       # Velocity dimension
gamma = -dv
C = 0.
qe = 1.
numerical_constants={"qe": qe, "C": C, "gamma": gamma}

# Create a mesh
box_length = 2 * jnp.pi / k
num_cells = 128 # small number for debugging
mesh = Mesh1D(box_length, num_cells)

# Create initial density distribution
initial_density = CosineNormal(alpha=alpha, k=k, dx=dx, dv=dv)

# Create neural network model
model = MLPScoreModel(dx, dv, hidden_dims=(64, ))

# Number of particles for simulation
num_particles = 100_000

# Define training configuration
training_config = {
    "batch_size": 64,
    "num_epochs": 0, # don't train NN when C=0
    "abs_tol": 1e-4,
    "learning_rate": 1e-3,
    "num_batch_steps": 0  # don't train NN when C=0
}

#%%
# SOLVE
print(f"N = {num_particles}, num_cells = {num_cells}, box_length = {box_length}, dx = {dx}, dv = {dv}")
solver = Solver(
    mesh=mesh,
    num_particles=num_particles,
    initial_density=initial_density,
    initial_nn=model,
    numerical_constants=numerical_constants,
    seed=seed
)
x0, v0, E0 = solver.x, solver.v, solver.E

box_length = mesh.box_lengths[0]
rho = qe*jax.vmap(lambda cell: jnp.mean(psi(solver.x - cell, solver.eta, box_length)))(mesh.cells())
plt.plot(mesh.cells(), rho - jnp.mean(rho), label='rho')
plt.plot(mesh.cells(), (jnp.roll(E0, 0) - jnp.roll(E0, 1)) / solver.eta, label='dE/dx')
plt.plot(mesh.cells(), E0, label='E')
plt.legend()
plt.show()

# Add training_config to solver (needed for the step method)
solver.training_config = training_config

# Train the initial model
train_initial_model(model, solver.x, solver.v, initial_density, training_config)

# Simulation parameters
final_time = 10.0 # set to 10 later
dt = 0.1
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
    if step % 20 == 0:
        print(f"Completed step {step+1}/{num_steps}, L2 norm of E: {e_l2_norms[step+1]:.6f}")
        plt.plot(mesh.cells(), E, label='E')
        plt.legend()
        plt.show()

# Save final state
solver.x, solver.v, solver.E = x, v, E

# Visualize results
visualize_results(solver, mesh, times, e_l2_norms)

#%%
import seaborn as sns
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# Plot L2 norm of E over time
axs[0].plot(times, e_l2_norms, label='L2 norm of E')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('||E||_2')
axs[0].set_title('L2 Norm of Electric Field')
axs[0].set_yscale('log')
axs[0].legend()

# Initial phase space KDE
kde1 = sns.kdeplot(
    x=x0[:, 0], y=v0[:, 0], fill=True, cmap='viridis', ax=axs[1], bw_adjust=0.5, levels=100, thresh=0.05
)
axs[1].set_xlabel('Position (x)')
axs[1].set_ylabel('Velocity (v1)')
axs[1].set_title(f'Initial Phase Space Density (KDE), t=0')
cbar1 = plt.colorbar(kde1.get_children()[0], ax=axs[1], label='Density')

# Final phase space KDE
kde2 = sns.kdeplot(
    x=solver.x[:, 0], y=solver.v[:, 0], fill=True, cmap='viridis', ax=axs[2], bw_adjust=0.5, levels=100, thresh=0.05
)
axs[2].set_xlabel('Position (x)')
axs[2].set_ylabel('Velocity (v1)')
axs[2].set_title(f'Final Phase Space Density (KDE), t={times[-1]:.2f}')
cbar2 = plt.colorbar(kde2.get_children()[0], ax=axs[2], label='Density')

plt.tight_layout()
plt.show()

# %%
# time step
from src.solver import evaluate_field_at_particles, update_velocities, update_positions, update_electric_field
import matplotlib.pyplot as plt
dt = 0.02

solver = Solver(
    mesh=mesh,
    num_particles=num_particles,
    initial_density=initial_density,
    initial_nn=model,
    numerical_constants=numerical_constants,
    seed=seed
)
x, v, E = solver.x, solver.v, solver.E

cells = mesh.cells()
eta = mesh.eta
C = numerical_constants["C"]
gamma = numerical_constants["gamma"]
box_length = mesh.box_lengths[0]

# 1. Evaluate electric field at particle positions
E_at_particles = evaluate_field_at_particles(x, cells, E, eta, box_length)

# 2. Update velocities (Vlasov + landau collision)
s = model(x, v)
v_new = update_velocities(v, E_at_particles, x, s, eta, C, gamma, dt, box_length)

# 3. Update positions using projected velocities
x_new = update_positions(x, v_new, dt, box_length)

# 4. Update electric field on the mesh
rho = evaluate_charge_density(x, cells, mesh.eta, box_length, qe=qe)
E1 = jnp.cumsum(rho - jnp.mean(rho)) * eta

#%%
# 1
indices = jnp.argsort(x[:, 0])
E_at_particles = evaluate_field_at_particles(x, cells, E, eta, box_length)

plt.plot(cells, E[:,0], label='E0 cells')
plt.plot(x[indices, 0], E_at_particles[indices, 0], label='E0 particles')
plt.plot(cells, E_new[:,0], label='E0 new')
plt.plot(cells, E_new[:,1], label='E1 new')
plt.legend()
plt.show()

#%%
# 2+3
jnp.min(x_new - (x + dt * v_new[:, :dx]) % box_length)

#%%

kernel_values = psi(cells[:, None] - x[None, :], eta, box_length)
jnp.sum(kernel_values>0)
E - dt * jnp.mean(kernel_values[:, :, None] * v, axis=1)