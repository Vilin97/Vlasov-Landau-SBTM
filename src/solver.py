#%%
import jax
import jax.numpy as jnp
from src.loss import explicit_score_matching_loss, implicit_score_matching_loss
import optax
from flax import nnx
from functools import partial
import jax.lax as lax

@jax.jit
def centered_mod(x, L):
    "centered_mod(x, L) in [-L/2, L/2]"
    return (x + L/2) % L - L/2

@jax.jit
def psi(x, eta, box_length):
    "psi_eta(x) = max(0, 1-|x|/eta) / eta."
    x = centered_mod(x, box_length)
    kernel = jnp.maximum(0.0, 1.0 - jnp.abs(x / eta))
    return kernel / eta

#%%
def train_initial_model(model, x, v, initial_density, training_config, verbose=False):
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
        if verbose:
            print(f"Epoch {epoch}: loss = {loss:.5f}")
        if loss < abs_tol:
            print(f"Stopping at epoch {epoch} with loss {loss :.5f} < {abs_tol}")
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
    """A(z) = C |z|^gamma (|z|² I_d − z⊗z)."""
    z_norm  = jnp.linalg.norm(z) + 1e-10
    factor  = C * z_norm ** gamma
    return factor * (jnp.eye(z.shape[0]) * z_norm**2 - jnp.outer(z, z))

@partial(jax.jit, static_argnames='num_cells')
def collision(x, v, s, eta, C, gamma, box_length, num_cells):
    """
    Q_i = (1/N) Σ_{|x_i−x_j|≤η} ψ(x_i−x_j) · A(v_i−v_j)(s_i−s_j)
          with the linear-hat kernel ψ of width η, periodic on [0,L].

    Complexity O(N η/L)  (exact for the hat kernel).
    """
    x = x[:,0]
    N, d = v.shape
    M    = num_cells

    # ---- bin & sort particles by cell --------------------------------------
    idx    = jnp.floor(x / eta).astype(jnp.int32) % M
    order  = jnp.argsort(idx)
    x_s, v_s, s_s, idx_s = x[order], v[order], s[order], idx[order]

    counts   = jnp.bincount(idx_s, length=M)
    cell_ofs = jnp.cumsum(jnp.concatenate([jnp.array([0]), counts[:-1]]))

    # ---- per-particle collision using lax.fori_loop (no dynamic slicing) ---
    def Q_single(i):
        xi, vi, si = x_s[i], v_s[i], s_s[i]
        cell       = idx_s[i]
        Q_i        = jnp.zeros(d)

        def loop_over_cell(c, acc):
            start = cell_ofs[c]
            end   = start + counts[c]

            def loop_over_particles(j, inner_acc):
                xj, vj, sj = x_s[j], v_s[j], s_s[j]
                w  = psi(xi - xj, eta, box_length)
                dv = vi - vj
                ds = si - sj
                inner_acc += w * jnp.dot(A(dv, C, gamma), ds)
                return inner_acc

            acc = lax.fori_loop(start, end, loop_over_particles, acc)
            return acc

        # neighbour cells that overlap the hat (periodic)
        for c in ((cell - 1) % M, cell, (cell + 1) % M):
            Q_i = loop_over_cell(c, Q_i)

        return Q_i / N

    Q_sorted = jax.vmap(Q_single)(jnp.arange(N))

    # ---- unsort back to original particle order ----------------------------
    rev = jnp.empty_like(order)
    rev = rev.at[order].set(jnp.arange(N))
    return Q_sorted[rev]

@jax.jit
def evaluate_charge_density(x, cells, eta, box_length, qe=1.0):
    """
    ρ_j = qe * box_length * ⟨ψ(x − cell_j)⟩   with ψ the same hat kernel.
    O(N) scatter-add instead of vmap over cells.
    """
    x = x[:,0]
    M      = cells.size
    idx_f  = x / eta - 0.5
    i0     = jnp.floor(idx_f).astype(jnp.int32) % M
    f      = idx_f - jnp.floor(idx_f)
    i1     = (i0 + 1) % M
    w0, w1 = 1.0 - f, f

    counts = (
        jnp.zeros(M)
          .at[i0].add(w0)
          .at[i1].add(w1)
    )
    return qe * box_length * counts / (x.size * eta)

@jax.jit
def evaluate_field_at_particles(x, cells, E, eta, box_length):
    """
    eta * Σ_j ψ(x_i − cell_j) E_j   (linear-hat kernel, periodic)
    Now O(N): two-point linear interpolation of E instead of a full sum.
    """
    x = x[:,0]
    M      = cells.size
    idx_f  = x / eta - 0.5
    i0     = jnp.floor(idx_f).astype(jnp.int32) % M
    f      = idx_f - jnp.floor(idx_f)
    i1     = (i0 + 1) % M
    return (1.0 - f) * E[i0] + f * E[i1]

@jax.jit
def update_electric_field(E, cells, x, v, eta, dt, box_length):
    """Works only with the triangular kernel."""
    x = x[:,0]
    M = cells.size
    idx_f = x / eta - 0.5
    i0    = jnp.floor(idx_f).astype(jnp.int32) % M
    f     = idx_f - jnp.floor(idx_f)
    i1    = (i0 + 1) % M
    w0, w1 = 1.0 - f, f
    J = (jnp.zeros(M)
          .at[i0].add(w0 * v[:, 0])
          .at[i1].add(w1 * v[:, 0])
        / (x.size * eta))
    return (E - dt * box_length * J).astype(E.dtype)

@jax.jit
def evaluate_electric_field(rho, eta):
    """Evaluate electric field on the mesh."""
    E = jnp.cumsum(rho - jnp.mean(rho)) * eta
    return E

@jax.jit
def update_positions(x, v, dt, box_length):
    """Update positions using velocities, modulo box_length."""
    dx = x.shape[-1]
    x = x + dt * v[:, :dx]
    return jnp.mod(x, box_length)

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
        assert mesh.eta < box_length/2, "eta must be smaller than half the box length"

        # 1) Sample particle positions and velocities
        key = jax.random.PRNGKey(seed)
        self.x, self.v = initial_density.sample(key, size=num_particles)

        # 2) Compute initial charge density
        qe = self.numerical_constants["qe"]
        rho = evaluate_charge_density(self.x, mesh.cells(), mesh.eta, box_length, qe=qe)

        # 3) Compute electric field
        self.E = evaluate_electric_field(rho, self.eta)
        
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
        num_cells = cells.size
        eta = self.eta
        C = self.numerical_constants["C"]
        gamma = self.numerical_constants["gamma"]
        box_length = self.mesh.box_lengths[0]
        
        # 1. Evaluate electric field at particle positions
        E_at_particles = evaluate_field_at_particles(x, cells, E, eta, box_length)
        
        # 2. Update velocities (Vlasov + landau collision)
        v_new = v.at[:, 0].add(dt * E_at_particles)
        if C != 0:
            s = self.score_model(x, v)
            v_new = v_new - dt * collision(x, v, s, eta, C, gamma, box_length, num_cells)
        
        # 3. Update positions using projected velocities
        x_new = update_positions(x, v_new, dt, box_length)
        
        # 4. Update electric field on the mesh
        E_new = update_electric_field(E, cells, x_new, v_new, eta, dt, box_length)
        
        # 5. Train score network
        if C != 0:
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
