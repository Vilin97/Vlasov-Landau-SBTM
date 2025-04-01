#%%
import jax
import jax.numpy as jnp
from src.mesh import Mesh1D
from src.density import Density
from src.score_model import create_mlp_score_model, create_resnet_score_model
from src.loss import explicit_score_matching_loss, implicit_score_matching_loss
import optax
from flax import nnx

@jax.jit
def psi(x: jnp.ndarray, eta: jnp.ndarray) -> jnp.ndarray:
    "psi_eta(x) = prod_i G(x_i/eta_i) / eta_i, where G(x) = max(0, 1-|x|)."
    return jnp.prod(jnp.maximum(0.0, 1.0 - jnp.abs(x / eta)) / eta, axis=-1)

#%%
key = jax.random.PRNGKey(0)
n, d = 4, 1  # Example dimensions
x = jax.random.normal(key, shape=(n, d))
eta = 1.
print(x)

mesh = Mesh1D(10, 4)
cells = mesh.cells()

rho = jax.vmap(lambda cell: jnp.mean(psi(x - cell, eta)))(cells)

print(mesh.laplacian())
print(rho)
phi1 = jnp.linalg.solve(mesh.laplacian(), jnp.mean(rho)-rho)
phi2, info=jax.scipy.sparse.linalg.cg(mesh.laplacian(), jnp.mean(rho)-rho)
print(phi1)
print(phi2)
print(phi1-jnp.mean(phi1))
print(phi2-jnp.mean(phi2))

#TODO: implement the electric field computation

#%%
class Solver:
    def __init__(
        self,
        mesh,
        num_particles,
        initial_density,
        initial_nn,
        numerical_constants={"qe": 1.0},
        eta=None,
        seed=0,
    ):
        """
        Initialize the solver with the given parameters and perform the steps:
        1. Sample particle positions and velocities from f_0
        2. Compute initial charge density
        3. Solve Poisson equation
        4. Compute E^0
        5. Train initial network
        """
        self.mesh = mesh
        self.num_particles = num_particles
        self.nn = initial_nn
        self.numerical_constants = numerical_constants
        
        # Set bandwidth of kernel equal to mesh width
        if eta is None:
            self.eta = mesh.eta
        else:
            self.eta = jnp.atleast_1d(jnp.asarray(eta))
            assert len(self.eta) == mesh.dim, "eta must have the same dimension as the mesh"

        # 1) Sample particle positions and velocities
        key = jax.random.PRNGKey(seed)
        self.x, self.v = initial_density.sample(key, size=num_particles)

        # 2) Compute initial charge density
        qe = self.numerical_constants["qe"]
        self.rho = qe*jax.vmap(lambda cell: jnp.mean(psi(x - cell, eta)))(cells)

        # 3) Solve Poisson equation
        phi, info = jax.scipy.sparse.linalg.cg(self.mesh.laplacian(), jnp.sum(self.rho) - rho)

        # 4) Compute E^0
        if mesh.boundary_condition == "periodic" and mesh.dim == 1:
            self.E = (jnp.roll(phi, -1) - jnp.roll(phi, 1)) / (2 * self.mesh.mesh_sizes[0])
        else:
            raise NotImplementedError("Non-periodic boundary conditions are not implemented.")

def train_initial_model(model, x, v, initial_density, training_config):
    """Train the initial neural network using the score of the initial density."""
    batch_size = training_config["batch_size"]
    num_epochs = training_config["num_epochs"]
    abs_tol    = training_config["abs_tol"]
    lr         = training_config["learning_rate"]
    optimizer  = optax.adamw(lr)
    score_vals = initial_density.score(x, v)
    loss_fn    = lambda model, batch: explicit_score_matching_loss(model, *batch)
    
    for epoch in range(num_epochs):
        loss = loss_fn(model, (x, v, score_vals))
        if loss < abs_tol:
            print(f"Early stopping at epoch {epoch} with loss {loss}")
            break
        
        perm = jax.random.permutation(jax.random.PRNGKey(epoch), len(x))
        x_shuffled, v_shuffled, s_shuffled = x[perm], v[perm], score_vals[perm]

        for i in range(0, len(x), batch_size):
            x_batch = x_shuffled[i:i + batch_size]
            v_batch = v_shuffled[i:i + batch_size]
            s_batch = s_shuffled[i:i + batch_size]
            batch = (x_batch, v_batch, s_batch)
            batch_loss = opt_step(model, optimizer, loss_fn, batch)

@nnx.jit(static_argnames='loss')
def opt_step(model, optimizer, loss, batch):
    """Perform one step of optimization"""
    loss_value, grads = nnx.value_and_grad(loss)(model, batch)
    optimizer.update(grads)
    return loss_value