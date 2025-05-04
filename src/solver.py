#%%
import jax
import jax.numpy as jnp
from src.loss import explicit_score_matching_loss, implicit_score_matching_loss
import optax
from flax import nnx

@jax.jit
def psi(x: jnp.ndarray, eta: jnp.ndarray, box_length) -> jnp.ndarray:
    "psi_eta(x) = prod_i G(x_i/eta_i) / eta_i, where G(x) = max(0, 1-|x|)."
    x = mod(x, box_length)
    kernel = jnp.maximum(0.0, 1.0 - jnp.abs(x / eta))
    # kernel = jax.scipy.stats.multivariate_normal.pdf(x / eta, 0, 1)
    return jnp.prod(kernel / eta, axis=-1)

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

def train_score_model(score_model, x_batch, v_batch, training_config):
    """
    Train the score network using implicit score matching loss.
    
    Args:
        x_batch: Particle positions of shape (num_particles, dx)
        v_batch: Particle velocities of shape (num_particles, dv)
        key: JAX random key or seed integer
        
    Returns:
        The list of batch losses during training.
    """
    # Extract training parameters from config
    batch_size = training_config["batch_size"]
    learning_rate = training_config["learning_rate"]
    num_batch_steps = training_config["num_batch_steps"]
    num_samples = x_batch.shape[0]
    div_mode = training_config.get("div_mode", "reverse")
    
    optimizer = nnx.Optimizer(score_model, optax.adamw(learning_rate))
    
    # Define loss function without div_mode parameter in the inner lambda
    def loss_fn(model, batch, key):
        return implicit_score_matching_loss(
            model, 
            batch[0], batch[1], 
            key=key, 
            div_mode=div_mode
        )
    
    step = 0
    batch_losses = []
    for epoch in range(num_batch_steps):
        # Generate a random key for this step
        epoch_key = jax.random.PRNGKey(epoch)
        
        # Shuffle data for each step
        perm = jax.random.permutation(epoch_key, num_samples)
        x_shuffled, v_shuffled = x_batch[perm], v_batch[perm]
        
        # Process mini-batches
        for i in range(0, num_samples, batch_size):
            step_key = jax.random.fold_in(epoch_key, i)
            x_mini = x_shuffled[i:i + batch_size]
            v_mini = v_shuffled[i:i + batch_size]
            mini_batch = (x_mini, v_mini)
            
            batch_loss = opt_step(score_model, optimizer, loss_fn, mini_batch, step_key)
            batch_losses.append(batch_loss)
            step += 1
            if step == num_batch_steps:
                return batch_losses

@nnx.jit(static_argnames='loss')
def opt_step(model, optimizer, loss, batch, key=None):
    """Perform one step of optimization"""
    if key is not None:
        loss_value, grads = nnx.value_and_grad(loss)(model, batch, key)
    else:
        loss_value, grads = nnx.value_and_grad(loss)(model, batch)
    optimizer.update(grads)
    return loss_value

@jax.jit
def A(z, C, gamma):
    "Collision kernel A(z) = C|z|^(γ)(I_d|z|^2 - z⊗z)"
    z_norm = jnp.linalg.norm(z)
    z_norm_safe = jnp.maximum(z_norm, 1e-10)
    z_norm_pow = z_norm_safe ** gamma
    z_outer = jnp.outer(z, z)
    I_scaled = jnp.eye(z.shape[0]) * (z_norm ** 2)
    return C * z_norm_pow * (I_scaled - z_outer)

@jax.jit
def collision(x, v, s, eta, C, gamma, box_length):
    "Collision operator Q(f,f) = ¹⁄ₙ ∑ ψ(x[p] - x[q]) A(v[p] - v[q]) * (s[p] - s[q])"
    def compute_single_collision(xp, vp, sp):
        collision_terms = jax.vmap(
            lambda xq, vq, sq: psi(xp - xq, eta, box_length) * jnp.dot(A(vp - vq, C, gamma), (sp - sq))
        )(x, v, s)
        return jnp.mean(collision_terms, axis=0)
    return jax.vmap(compute_single_collision)(x, v, s)

# TODO: [speed] since cells are eta apart, the array psi(x_i - cells, eta) has *exactly* 2 non-zero entries
@jax.jit
def evaluate_field_at_particles(x, cells, E, eta, box_length):
    """Evaluate electric field at particle positions."""
    return jax.vmap(lambda x_i: eta * jnp.sum(psi(x_i - cells, eta, box_length)[:, None] * E, axis=0))(x)

@jax.jit
def update_velocities(v, E_at_particles, x, s, eta, C, gamma, dt, box_length):
    """Update velocities using electric field and collision term."""
    collision_term = collision(x, v, s, eta, C, gamma, box_length)
    return v + dt * (E_at_particles - collision_term)

@jax.jit
def update_positions(x, v, dt, box_length):
    """Update positions using velocities, modulo box_length."""
    dx = x.shape[-1]
    if v.shape[-1] == dx:
        x = x + dt * v
    else:
        x = x + dt * v[..., :dx]
    return mod(x, box_length)

@jax.jit
def mod(x, box_length):
    "compute x modulo box_length"
    mask = 1. * (x < 0) - 1 * (x > box_length)
    return x + mask * box_length

@jax.jit
def update_electric_field(E, cells, x, v, eta, dt, box_length):
    """Update electric field on the mesh."""
    kernel_values = psi(cells[:, None] - x[None, :], eta, box_length)
    return E - dt * jnp.mean(kernel_values[:, :, None] * v, axis=1).reshape(E.shape)

@jax.jit
def compute_electric_field(v, rho, eta):
    """Compute electric field on the mesh."""
    E1 = jnp.cumsum(rho - jnp.mean(rho)) * eta
    E = jnp.zeros((E1.shape[0], v.shape[-1]))
    return E.at[:, 0].set(E1)

@jax.jit
def evaluate_charge_density(x, cells, eta, box_length, qe=1):
    rho = qe * jax.vmap(lambda cell: jnp.mean(psi(x - cell, eta, box_length)))(cells)
    return rho

#%%
class Solver:
    def __init__(
        self,
        mesh,
        num_particles,
        initial_density,
        initial_nn,
        numerical_constants,
        eta=None,
        seed=0,
        training_config=None,
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
            training_config: Configuration for training the score model
        """
        self.mesh = mesh
        self.num_particles = num_particles
        self.score_model = initial_nn
        self.numerical_constants = numerical_constants
        self.training_config = training_config
        box_length = self.mesh.box_lengths[0]
        
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
        rho = evaluate_charge_density(self.x, mesh.cells(), mesh.eta, box_length, qe=qe)

        # 3) Solve Poisson equation
        # phi, info = jax.scipy.sparse.linalg.cg(self.mesh.laplacian(), jnp.mean(rho) - rho)

        # 4) Compute electric field
        if mesh.boundary_condition == "periodic" and mesh.dim == 1:
            # E1 = -(jnp.roll(phi, -1) - jnp.roll(phi, 1)) / (2 * self.mesh.eta[0])
            E1 = jnp.cumsum(rho - jnp.mean(rho)) * self.eta
            E = jnp.zeros((E1.shape[0], self.v.shape[-1]))
            self.E = E.at[:, 0].set(E1)
        else:
            raise NotImplementedError("Non-periodic boundary conditions are not implemented.")
        
    def step(self, x, v, E, dt):
        """
        Perform a single time step of the simulation.
        
        Args:
            x: Particle positions
            v: Particle velocities
            E: Electric field
            dt: Time step size
            
        Returns:
            Updated particle positions, velocities, and electric field
        """
        cells = self.mesh.cells()
        eta = self.eta
        C = self.numerical_constants["C"]
        gamma = self.numerical_constants["gamma"]
        box_length = self.mesh.box_lengths[0]
        
        # 1. Evaluate electric field at particle positions
        E_at_particles = evaluate_field_at_particles(x, cells, E, eta, box_length)
        
        # 2. Update velocities (Vlasov + landau collision)
        s = self.score_model(x, v)
        v_new = update_velocities(v, E_at_particles, x, s, eta, C, gamma, dt, box_length)
        
        # 3. Update positions using projected velocities
        x_new = update_positions(x, v_new, dt, box_length)
        
        # 4. Update electric field on the mesh
        E_new = update_electric_field(E, cells, x, v, eta, dt, box_length)
        # qe = self.numerical_constants["qe"]
        # rho = evaluate_charge_density(x, self.mesh.cells(), self.mesh.eta, box_length, qe=qe)
        # E1 = jnp.cumsum(rho - jnp.mean(rho)) * self.eta
        # E_new = jnp.zeros((E1.shape[0], self.v.shape[-1]))
        # E_new = E_new.at[:, 0].set(E1)
        
        # 5. Train score network
        train_score_model(self.score_model, x_new, v_new, self.training_config)
        
        return x_new, v_new, E_new
    
    def solve(self, final_time, dt):
        """
        Solve the Maxwell-Vlasov-landau system using SBTM.
        
        Args:
            final_time: Final simulation time
            dt: Time step size
            key: JAX random key or seed integer
            
        Returns:
            Particle states, electric field, and score model at final time
        """
        # Calculate number of time steps
        num_steps = int(final_time / dt)
        
        # Initial particle states and field are already set in __init__
        x = self.x
        v = self.v
        E = self.E
        
        for step_idx in range(num_steps):
            # Optional logging
            if step_idx % 10 == 0:
                print(f"Step {step_idx}/{num_steps}")
            
            # Perform one simulation step
            x, v, E = self.step(x, v, E, dt)
            
        # Save final state
        self.x, self.v, self.E = x, v, E
        
        return self.x, self.v, self.E, self.score_model
