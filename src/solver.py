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
# learn to interpolate electric field between cells and particles
from scipy.stats import gaussian_kde
import jax.random
import matplotlib.pyplot as plt

# Define parameters
n, d = 100, 1
m = 40
key = jax.random.PRNGKey(0)
eta = 10/m

# Create cells array
cells = jnp.linspace(eta/2, 10 - eta/2, m).reshape(m, d)

# Create particles array
x = jax.random.normal(key, shape=(n, d)) * 1 + 5
x = jnp.sort(x, axis=0)

# Plot the KDE of x

kde = gaussian_kde(x.flatten())
x_vals = jnp.linspace(x.min(), x.max(), 500)
kde_vals = kde(x_vals)

plt.plot(x_vals, kde_vals, label='KDE of x')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

# Charge density
rho_p = jnp.ones((n, 1)) / n
plt.plot(x, rho_p, label='initial density, uniform')

for i in range(4):
    rho_c = jax.vmap(lambda c: jnp.mean(psi(c - x, eta) * rho_p))(cells)
    plt.plot(cells, rho_c, label=f'density at cells {i+1}')
    rho_p = jax.vmap(lambda x: jnp.mean(psi(x - cells, eta) * rho_c))(x)
    plt.plot(x, rho_p, label=f'density at particles {i+1}')

plt.legend()

#%%
# Compute the collision term
from src.density import CosineNormal
import time
dv = 2
density = CosineNormal(k = 4, dv=dv)
key = jax.random.PRNGKey(0)
n = 200
x,v = density.sample(key, size=n)
# s = density.score(x,v)
s = jax.random.uniform(key, shape=(n, dv))
eta = 1.

@jax.jit
def A(z,C=1,gamma=-3):
    z_norm = jnp.linalg.norm(z)
    z_norm_safe = jnp.maximum(z_norm, 1e-10)
    z_norm_pow = z_norm_safe ** (-gamma)
    z_outer = jnp.outer(z, z)
    I_scaled = jnp.eye(z.shape[0]) * (z_norm ** 2)
    return C * z_norm_pow * (I_scaled - z_outer)

def compute_collision_term_simple(xp, vp, sp, x, v, s, eta, C=1, gamma=-3):
    N = len(x)
    collision_term = jnp.zeros_like(vp)

    for q in range(N):
        # Compute psi_h(xp - xq)
        diff_x = xp - x[q]
        psi_h = 1.0
        for i in range(len(diff_x)):
            psi_h *= max(0.0, 1.0 - abs(diff_x[i] / eta)) / eta

        # Compute A(vp - vq)
        diff_v = vp - v[q]
        z_norm = max((sum(diff_v[i] ** 2 for i in range(len(diff_v)))) ** 0.5, 1e-10)
        z_norm_pow = z_norm ** (-gamma)
        z_outer = [[diff_v[i] * diff_v[j] for j in range(len(diff_v))] for i in range(len(diff_v))]
        I_scaled = [[(z_norm ** 2 if i == j else 0.0) for j in range(len(diff_v))] for i in range(len(diff_v))]
        A_matrix = [[C * z_norm_pow * (I_scaled[i][j] - z_outer[i][j]) for j in range(len(diff_v))] for i in range(len(diff_v))]

        # Compute (sp - sq)
        score_diff = [sp[i] - s[q][i] for i in range(len(sp))]

        # Compute A(vp - vq) * (sp - sq)
        A_dot_score_diff = [sum(A_matrix[i][j] * score_diff[j] for j in range(len(score_diff))) for i in range(len(score_diff))]

        # Accumulate the weighted term
        collision_term += psi_h * jnp.array(A_dot_score_diff)

    return collision_term / N

def compute_single_collision(xp, vp, sp):
    collision_terms = jax.vmap(
        lambda xq, vq, sq: psi(xp - xq, eta) * jnp.dot(A(vp - vq), (sp - sq))
    )(x, v, s)
    return jnp.mean(collision_terms, axis=0)

@jax.jit
def collision(x, v, s, eta):
    "Collision operator Q(f,f) = ¹⁄ₙ ∑ ψ(x[p] - x[q]) A(v[p] - v[q]) * (s[p] - s[q])"
    def compute_single_collision(xp, vp, sp):
        collision_terms = jax.vmap(
            lambda xq, vq, sq: psi(xp - xq, eta) * jnp.dot(A(vp - vq), (sp - sq))
        )(x, v, s)
        return jnp.mean(collision_terms, axis=0)
    return jax.vmap(compute_single_collision)(x, v, s)


xp, vp, sp = x[0], v[0], s[0]
xq, vq, sq = x[1], v[1], s[1]

psi(xp - xq, eta) * jnp.dot(A(vp - vq), (sp - sq))
compute_single_collision(xp, vp, sp)

# Measure time for the loop
start_time = time.time()
for (xp, vp, sp) in zip(x, v, s):
    compute_single_collision(xp, vp, sp)
loop_time = time.time() - start_time
print(f"Loop time: {loop_time:.6f} seconds")

# Measure time for vmap
jax.vmap(compute_single_collision)(x, v, s)
start_time = time.time()
jax.vmap(compute_single_collision)(x, v, s)
vmap_time = time.time() - start_time
print(f"vmap time: {vmap_time:.6f} seconds")


# def collision(xp, vp, sp, eta):
#     return psi(xp, eta) * A(vp) * sp


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
def collision_kernel_A(C, gamma, z):
        """
        Compute the collision kernel A(z) = C|z|^(-γ)(I_d|z|^2 - z⊗z)
        
        Args:
            z: Relative velocity vector (v - v')
            
        Returns:
            Collision kernel matrix A(z)
        """
        
        # Compute |z|
        z_norm = jnp.linalg.norm(z)
        
        # Avoid division by zero
        z_norm_safe = jnp.maximum(z_norm, 1e-10)
        
        # Compute |z|^(-γ)
        z_norm_pow = z_norm_safe ** (-gamma)
        
        # Compute outer product z⊗z
        z_outer = jnp.outer(z, z)
        
        # Compute identity matrix scaled by |z|^2
        I_scaled = jnp.eye(z.shape[0]) * (z_norm ** 2)
        
        # Compute A(z) = C|z|^(-γ)(I_d|z|^2 - z⊗z)
        return C * z_norm_pow * (I_scaled - z_outer)

def collision(x, v, s, eta):
    "Collision operator Q(f,f) = ¹⁄ₙ ∑ ψ(x[p] - x[q]) A(v[p] - v[q]) * (s[p] - s[q])"
    def compute_single_collision(xp, vp, sp):
        collision_terms = jax.vmap(
            lambda xq, vq, sq: psi(xp - xq, eta) * jnp.dot(A(vp - vq), (sp - sq))
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
    
    def collision_term(self, x, v, s_x_v, x_other, v_other, s_x_v_other):
        """
        Compute the Landau collision term for a particle using 
        the collision kernel A(v-v') = C|v-v'|^(-γ)(I_d|v-v'|^2 - (v-v')⊗(v-v'))
        
        Args:
            x: Position of the particle
            v: Velocity of the particle
            s_x_v: Score at (x, v)
            x_other: Positions of other particles
            v_other: Velocities of other particles
            s_x_v_other: Scores at other particles (x_other, v_other)
            
        Returns:
            Collision term contribution to velocity update
        """
        # Calculate spatial kernel between particles
        kernel_values = psi(x - x_other, self.eta)
        
        # Calculate score differences
        score_diff = s_x_v - s_x_v_other
        
        # Calculate collision kernel for each particle pair
        def apply_collision_kernel(v_diff, score_diff):
            A_matrix = self.collision_kernel_A(v_diff)
            return jnp.dot(A_matrix, score_diff)
        
        v_diffs = v - v_other
        weighted_collision = jax.vmap(apply_collision_kernel)(v_diffs, score_diff)
        
        # Calculate the collision term with proper weighting
        return jnp.mean(kernel_values[:, None] * weighted_collision, axis=0)
    
    def projection_dx_dv(self, v):
        """
        Project velocity from dv dimensions to dx dimensions.
        
        Args:
            v: Velocity vector of shape (dv,)
            
        Returns:
            Projected velocity of shape (dx,)
        """
        # If dx == dv, return v as is, otherwise take first dx components
        return v[:self.mesh.dim]
    
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
        # Convert integer key to PRNG key if needed
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)
        
        # Extract parameters from config
        score_update_steps = self.training_config["score_update_steps"]
        
        # Calculate number of time steps
        num_steps = int(final_time / dt)
        
        # Initial particle states and field are already set in __init__
        x_t = self.x
        v_t = self.v
        E_t = self.E
        
        for t in range(num_steps):
            # Optional logging
            if t % 10 == 0:
                print(f"Step {t}/{num_steps}")
            
            # 1. Evaluate electric field at particle positions
            E_at_particles = jax.vmap(lambda x: jnp.mean(psi(x - cells, eta) * self.E))(x)
            
            # 2. Compute scores at current particle positions
            scores_t = self.score_model(x_t, v_t)
            
            # 3. Update velocities (Vlasov + Landau collision)
            # Compute collision term for each particle
            collision_terms = jax.vmap(
                lambda x, v, s, x_o, v_o, s_o: self.collision_term(x, v, s, x_o, v_o, s_o)
            )(x_t, v_t, scores_t, x_t, v_t, scores_t)
            
            # Update velocities (collision strength C is now incorporated in the collision kernel)
            v_next = v_t + dt * (E_at_particles - collision_terms)
            
            # 4. Update positions using projected velocities
            position_updates = jax.vmap(self.projection_dx_dv)(v_next)
            x_next = x_t + dt * position_updates
            
            # 5. Update electric field on the mesh
            # Compute the field update term
            field_update = jnp.zeros_like(E_t)
            for cell_idx, cell in enumerate(self.mesh.cells()):
                kernel_values = jax.vmap(lambda x: psi(cell - x, self.eta))(x_t)
                field_update = field_update.at[cell_idx].set(
                    -dt * jnp.mean(kernel_values[:, None] * v_t, axis=0)[0]
                )
            
            E_next = E_t + field_update
            
            # 6. Train score network
            step_key = jax.random.fold_in(key, t)
            self.train_score_net(x_t, v_t, key=step_key)
            
            # Update state for next iteration
            x_t, v_t, E_t = x_next, v_next, E_next
        
        # Save final state
        self.x, self.v, self.E = x_t, v_t, E_t
        
        return self.x, self.v, self.E, self.score_model
