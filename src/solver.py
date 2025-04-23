#%%
import jax
import jax.numpy as jnp
from src.loss import explicit_score_matching_loss, implicit_score_matching_loss
import optax
from flax import nnx

@jax.jit
def psi(x: jnp.ndarray, eta: jnp.ndarray) -> jnp.ndarray:
    "psi_eta(x) = prod_i G(x_i/eta_i) / eta_i, where G(x) = max(0, 1-|x|)."
    return jnp.prod(jnp.maximum(0.0, 1.0 - jnp.abs(x / eta)) / eta, axis=-1)

#%%

def train_initial_model(model, x, v, initial_density, training_config):
    """Train the initial neural network using the score of the initial density."""
    batch_size = training_config["batch_size"]
    num_epochs = training_config["num_epochs"]
    abs_tol    = training_config["abs_tol"]
    lr         = training_config["learning_rate"]
    
    optimizer  = nnx.Optimizer(model, optax.adamw(lr))
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

@jax.jit
def A(z, C, gamma):
    "Collision kernel A(z) = C|z|^(-γ)(I_d|z|^2 - z⊗z)"
    z_norm = jnp.linalg.norm(z)
    z_norm_safe = jnp.maximum(z_norm, 1e-10)
    z_norm_pow = z_norm_safe ** (-gamma)
    z_outer = jnp.outer(z, z)
    I_scaled = jnp.eye(z.shape[0]) * (z_norm ** 2)
    return C * z_norm_pow * (I_scaled - z_outer)

@jax.jit
def collision(x, v, s, eta, C, gamma):
    "Collision operator Q(f,f) = ¹⁄ₙ ∑ ψ(x[p] - x[q]) A(v[p] - v[q]) * (s[p] - s[q])"
    def compute_single_collision(xp, vp, sp):
        collision_terms = jax.vmap(
            lambda xq, vq, sq: psi(xp - xq, eta) * jnp.dot(A(vp - vq, C, gamma), (sp - sq))
        )(x, v, s)
        return jnp.mean(collision_terms, axis=0)
    return jax.vmap(compute_single_collision)(x, v, s)

class Solver:
    def __init__(
        self,
        mesh,
        num_particles,
        initial_density,
        initial_nn,
        numerical_constants={"qe": 1.0, "C": 1.0, "gamma": 3},
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
        
        Args:
            mesh: Mesh object
            num_particles: Number of particles
            initial_density: Initial density function
            initial_nn: Initial neural network model, pre-trained separately
            numerical_constants: Dictionary of numerical constants including:
                - qe: Charge
                - C: Collision strength coefficient
                - gamma: Collision kernel exponent
            eta: Bandwidth parameter
            seed: Random seed
        """
        self.mesh = mesh
        self.num_particles = num_particles
        self.score_model = initial_nn
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
        rho = qe*jax.vmap(lambda cell: jnp.mean(psi(self.x - cell, self.eta)))(mesh.cells())

        # 3) Solve Poisson equation
        phi, info = jax.scipy.sparse.linalg.cg(self.mesh.laplacian(), jnp.sum(rho) - rho)

        # 4) Compute electric field
        if mesh.boundary_condition == "periodic" and mesh.dim == 1:
            self.E = (jnp.roll(phi, -1) - jnp.roll(phi, 1)) / (2 * self.mesh.eta[0])
        else:
            raise NotImplementedError("Non-periodic boundary conditions are not implemented.")
        
    def train_score_net(self, x_batch, v_batch, key=0):
        """
        Train the score network using implicit score matching loss.
        
        Args:
            x_batch: Particle positions of shape (num_particles, dx)
            v_batch: Particle velocities of shape (num_particles, dv)
            key: JAX random key or seed integer
            
        Returns:
            Updated score model
        """
        # Convert integer key to PRNG key if needed
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)
        
        # Extract training parameters from config
        batch_size = self.training_config["batch_size"]
        learning_rate = self.training_config["learning_rate"]
        num_steps = self.training_config["score_update_steps"]
        
        optimizer = nnx.Optimizer(self.score_model, optax.adamw(learning_rate))
        loss_fn = lambda model, batch, key: implicit_score_matching_loss(
            lambda x, v: model(x, v), 
            batch[0], batch[1], 
            key=key, div_mode='reverse'
        )
        
        # Create batches
        num_samples = x_batch.shape[0]
        
        for step in range(num_steps):
            # Generate a random key for this step
            step_key = jax.random.fold_in(key, step)
            
            # Shuffle data for each step
            perm = jax.random.permutation(step_key, num_samples)
            x_shuffled, v_shuffled = x_batch[perm], v_batch[perm]
            
            # Process mini-batches
            for i in range(0, num_samples, batch_size):
                x_mini = x_shuffled[i:i + batch_size]
                v_mini = v_shuffled[i:i + batch_size]
                mini_batch = (x_mini, v_mini)
                
                # Perform optimization step
                batch_loss = opt_step(self.score_model, optimizer, 
                                     lambda m, b: loss_fn(m, b, step_key), 
                                     mini_batch)
        
        return self.score_model
    
    def solve(self, final_time, dt, key=0):
        """
        Solve the Maxwell-Vlasov-Landau system using SBTM.
        
        Args:
            final_time: Final simulation time
            dt: Time step size
            key: JAX random key or seed integer
            
        Returns:
            Particle states, electric field, and score model at final time
        """
        
        dx = self.x.shape[-1]
        if v.shape[-1] == dx:
            projection_onto_dx = lambda v: v
        else:
            projection_onto_dx = lambda v: v[..., :dx]
        
        # Convert integer key to PRNG key if needed
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)
        
        # Extract parameters from config
        score_update_steps = self.training_config["score_update_steps"]
        
        # Calculate number of time steps
        num_steps = int(final_time / dt)
        
        # Initial particle states and field are already set in __init__
        x = self.x
        v = self.v
        E = self.E
        cells = self.mesh.cells()
        eta = self.eta
        C = self.numerical_constants["C"]
        gamma = self.numerical_constants["gamma"]
        
        for step in range(num_steps):
            # Optional logging
            if step % 10 == 0:
                print(f"Step {step}/{num_steps}")
            
            # 1. Evaluate electric field at particle positions
            E_at_particles = jax.vmap(lambda x: jnp.mean(psi(x - cells, eta) * E))(x)
            
            # 2. Update velocities (Vlasov + Landau collision)
            s = self.score_model(x, v)
            collision_term = collision(x, v, s, eta, C, gamma)
            v_next = v + dt * (E_at_particles - collision_term)
            
            # 3. Update positions using projected velocities
            x_next = x + dt * projection_onto_dx(v_next)
            
            # TODO this step and below are still to be implemented
            # 4. Update electric field on the mesh
            field_update = jnp.zeros_like(E_t)
            for cell_idx, cell in enumerate(self.mesh.cells()):
                kernel_values = jax.vmap(lambda x: psi(cell - x, self.eta))(x)
                field_update = field_update.at[cell_idx].set(
                    -dt * jnp.mean(kernel_values[:, None] * v, axis=0)[0]
                )
            
            kernel_values = psi(cells[:, None] - x[None, :], self.eta)
            field_update = -dt * jnp.mean(kernel_values[:, :, None] * v, axis=1)[:, 0]
            E_next = E_t + field_update
            
            # 5. Train score network
            step_key = jax.random.fold_in(key, t)
            self.train_score_net(x_t, v_t, key=step_key)
            
            # Update state for next iteration
            x_t, v_t, E_t = x_next, v_next, E_next
        
        # Save final state
        self.x, self.v, self.E = x_t, v_t, E_t
        
        return self.x, self.v, self.E, self.score_model

#%%
import jax.numpy as jnp

n_cells = 3
n_x = 5
d = 2
eta = jnp.array([1.])

key = jax.random.PRNGKey(0)
key1, key2, key3 = jax.random.split(key, 3)
cells = jax.random.normal(key1, (n_cells, d))
x = jax.random.normal(key2, (n_x, d))
v = jax.random.normal(key3, (n_x, d))

diff = cells[:, None] - x[None, :]
kernel_values = psi(diff, eta)
E = jnp.mean(kernel_values[:,:,None] * v, axis=1)

i = 0 # cell index
j = 1 # dimension index
# comput ecoordinate i,j
print(E[i,j])
sum((psi(cells[i,:] -  x[k,:], eta)) * v[k,j] for k in range(n_x)) / n_x

# %%
k = 2 # particle index
print(diff[i,k,j])
print(cells[i,j] -  x[k,j])

print(psi(cells[i,j] -  x[k,j], eta))
print(psi(diff[i,k,j], eta))

print(kernel_values[i,k])
print(psi(cells[i,:] -  x[k,:], eta))

print(kernel_values[i,k] * v[k,j])
print(psi(cells[i,:] -  x[k,:], eta) * v[k,j])

print((kernel_values[:,:,None] * v)[i,k,j])
print(psi(cells[i,:] -  x[k,:], eta) * v[k,j])

print(E[i,j])
print(sum([psi(cells[i,:] -  x[k,:], eta) * v[k,j] for k in range(n_x)]) / n_x)
