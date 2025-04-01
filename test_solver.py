#%%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx

from src.mesh import Mesh1D
from src.density import CosineNormal
from src.score_model import create_mlp_score_model
from src.solver import Solver, train_initial_model

#%%
import optax
model = create_mlp_score_model(hidden_dims=(64,))
nnx.Optimizer(model, optax.adamw(1e-3))
#TODO: switch score models to inherit from nnx.Module to make the above work

#%%
def main():
    # Set random seed for reproducibility
    seed = 42
    key = jax.random.PRNGKey(seed)
    
    # Create a mesh
    box_length = 2 * jnp.pi
    num_cells = 32
    mesh = Mesh1D(box_length, num_cells)
    
    # Create initial density distribution
    alpha = 0.1  # Perturbation strength
    k = 1.0      # Wave number
    dv = 1       # Velocity dimension
    initial_density = CosineNormal(alpha=alpha, k=k, dx=1, dv=dv)
    
    # Create neural network model
    model = create_mlp_score_model(hidden_dims=(64, 64, 64))
    
    # Number of particles for simulation
    num_particles = 10
    
    # Initialize the solver
    print("Initializing solver...")
    solver = Solver(
        mesh=mesh,
        num_particles=num_particles,
        initial_density=initial_density,
        initial_nn=model,
        seed=seed
    )
    
    # Train the model
    training_config = {
        "batch_size": 64,
        "num_epochs": 10,
        "abs_tol": 1e-4,
        "learning_rate": 1e-3
    }
    print("Training initial model...")
    train_initial_model(model, solver.x, solver.v, initial_density, training_config)
    
    # Test the trained model
    print("Testing trained model...")
    # Sample some test points
    test_key = jax.random.PRNGKey(seed + 1)
    x_test, v_test = initial_density.sample(test_key, size=100)
    
    # Get the scores from the model and analytical solution
    model_scores = jax.vmap(lambda x, v: model(x, v))(x_test, v_test)
    true_scores = initial_density.score(x_test, v_test)
    
    # Compute mean squared error
    mse = jnp.mean((model_scores - true_scores)**2)
    print(f"Mean squared error: {mse:.6f}")
    
    # Visualize results
    visualize_results(mesh, solver, x_test, v_test, model_scores, true_scores)
    
def visualize_results(mesh, solver, x_test, v_test, model_scores, true_scores):
    """Visualize the results of the solver and model training."""
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Particle phase space
    plt.subplot(2, 2, 1)
    plt.scatter(solver.x.flatten(), solver.v.flatten(), s=1, alpha=0.3)
    plt.title('Particle Phase Space')
    plt.xlabel('Position (x)')
    plt.ylabel('Velocity (v)')
    
    # Plot 2: Charge density
    plt.subplot(2, 2, 2)
    x_cells = mesh.cells().flatten()
    plt.plot(x_cells, solver.rho)
    plt.title('Charge Density')
    plt.xlabel('Position (x)')
    plt.ylabel('Density (ρ)')
    
    # Plot 3: Electric field
    plt.subplot(2, 2, 3)
    plt.plot(x_cells, solver.E)
    plt.title('Electric Field')
    plt.xlabel('Position (x)')
    plt.ylabel('Electric Field (E)')
    
    # Plot 4: Score model comparison (for a single velocity dimension)
    plt.subplot(2, 2, 4)
    plt.scatter(true_scores.flatten(), model_scores.flatten(), s=2, alpha=0.5)
    max_val = max(jnp.max(jnp.abs(true_scores)), jnp.max(jnp.abs(model_scores)))
    plt.plot([-max_val, max_val], [-max_val, max_val], 'r--')
    plt.title('Score Comparison')
    plt.xlabel('True Score')
    plt.ylabel('Model Score')
    
    plt.tight_layout()
    plt.savefig('solver_results.png')
    plt.show()

if __name__ == "__main__":
    main()
