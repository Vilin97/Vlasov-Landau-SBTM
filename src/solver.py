#%%
import jax
import jax.numpy as jnp
from src.mesh import Mesh1D
from src.density import Density
from src.score_model import create_mlp_score_model, create_resnet_score_model
from src.loss import explicit_score_matching_loss, implicit_score_matching_loss
import optax


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
        training_config,
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
        self.rho = self.compute_charge_density(self.x)

        # 3) Solve Poisson equation
        phi = self.solve_poisson(self.rho - jnp.sum(self.rho))

        # 4) Compute E^0
        self.E = self.compute_electric_field(phi)

        # 5) Train initial_nn using the score of f_0
        self.train_initial_nn()

    def compute_charge_density(self, x):
        """Compute the charge density on the mesh."""
        qe = self.numerical_constants["qe"]
        rho = qe*jax.vmap(lambda cell: jnp.mean(psi(x - cell, eta)))(cells)
        return rho

    def solve_poisson(self, rho):
        """Solve the Poisson equation using the Laplacian matrix."""
        phi, info = jax.scipy.sparse.linalg.cg(self.mesh.laplacian(), -rho)
        return phi

    def compute_electric_field(self, phi):
        """Compute the electric field from the potential."""
        if mesh.boundary_condition == "periodic":
            E = (jnp.roll(phi, -1) - jnp.roll(phi, 1)) / (2 * self.mesh.mesh_sizes[0])
            return E
        else:
            raise NotImplementedError("Non-periodic boundary conditions are not implemented.")

    def train_initial_nn(self):
        # TODO
        """Train the initial neural network using the score of the initial density."""
        key = jax.random.PRNGKey(self.training_config["random_seed"])
        true_scores = self.initial_density.score(self.x, self.v)

        # Initialize optimizer
        optimizer = optax.adam(self.training_config["learning_rate"])
        params = self.initial_nn.init(key, self.x, self.v)
        opt_state = optimizer.init(params)

        for _ in range(self.initial_steps):

            def loss_fn(params):
                def model_fn(x, v):
                    return self.initial_nn.apply(params, x, v)

                return explicit_score_matching_loss(model_fn, self.x, self.v, true_scores)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

        self.s = params
