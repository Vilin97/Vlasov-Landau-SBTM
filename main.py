import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
import flax.linen as nn
import optax
from functools import partial
from typing import Tuple, List, Callable, Dict, Any

# Set precision and platform
jax.config.update("jax_enable_x64", True)  # Use double precision

# ============================
# Score Network Architecture
# ============================
class ScoreNetwork(nn.Module):
    features: List[int]
    
    @nn.compact
    def __call__(self, x_v):
        """
        Input: concatenated position and velocity vectors
        Output: score function (gradient of log density)
        """
        z = x_v
        for i, feat in enumerate(self.features):
            z = nn.Dense(feat)(z)
            if i < len(self.features) - 1:
                z = nn.tanh(z)
        return z

# ============================
# Helper Functions
# ============================
@jit
def psi_eta(x_diff, eta):
    """Gaussian kernel with width eta"""
    return jnp.exp(-0.5 * jnp.sum((x_diff / eta)**2)) / (jnp.sqrt(2 * jnp.pi) * jnp.prod(eta))

@jit
def compute_A_matrix(z, C, gamma):
    """
    Compute the matrix A(z) = C * |z|^(-gamma) * (I_d|z|^2 - z⊗z)
    """
    z_norm = jnp.linalg.norm(z)
    z_norm = jnp.where(z_norm < 1e-10, 1e-10, z_norm)  # Avoid division by zero
    
    identity = jnp.eye(z.shape[0])
    outer_prod = jnp.outer(z, z)
    
    return C * (z_norm ** (-gamma)) * (identity * (z_norm ** 2) - outer_prod)

# Project velocity to spatial dimensions
@jit
def project_velocity(v, d_x, d_v):
    """Project velocity vector from R^d_v to R^d_x"""
    return v[:d_x]

# ============================
# Poisson Solver
# ============================
@jit
def build_1d_laplacian(N, eta):
    """Build 1D discrete Laplacian matrix"""
    diag = -2.0 * jnp.ones(N)
    off_diag = jnp.ones(N-1)
    lap = jnp.diag(diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
    # Apply periodic boundary conditions
    lap = lap.at[0, N-1].set(1.0)
    lap = lap.at[N-1, 0].set(1.0)
    return lap / (eta**2)

@jit
def solve_poisson_equation(rho, rho_ion, laplacian):
    """
    Solve the Poisson equation -Δϕ = ρ - ρ_ion
    """
    # Ensure compatibility condition: ∫(ρ - ρ_ion) = 0
    rho_mean = jnp.mean(rho)
    rho_adj = rho - rho_mean + rho_ion
    
    # Solve linear system
    phi = jnp.linalg.solve(laplacian, -(rho_adj))
    return phi

@jit
def compute_electric_field_from_phi(phi, eta, d_x):
    """Compute electric field E = -∇ϕ using central differences"""
    # For 1D
    if d_x == 1:
        E = -(jnp.roll(phi, -1) - jnp.roll(phi, 1)) / (2 * eta)
        return E.reshape(-1, 1)  # Return as column vector
    
    
    # For higher dimensions, would need to implement

# ============================
# Score Network Training
# ============================
@jit
def dsm_loss(score_fn, particles_x, particles_v):
    """
    Compute the Denoising Score Matching loss
    L(s) = (1/N) ∑_q (|s(x_q, v_q)|^2 + 2∇·s(x_q, v_q))
    """
    xv = jnp.concatenate([particles_x, particles_v], axis=1)
    
    # Compute |s(x, v)|^2 term
    scores = vmap(score_fn)(xv)
    norm_term = jnp.mean(jnp.sum(scores**2, axis=1))
    
    # Compute divergence term using Hutchinson's estimator
    # This is an approximation of the trace of the Jacobian
    def div_estimator(params, xv, eps):
        def score_dot_eps(xv):
            return jnp.sum(score_fn(params, xv) * eps)
        return grad(score_dot_eps)(xv)
    
    key = random.PRNGKey(0)  # Should be refreshed in practice
    eps = random.normal(key, shape=xv.shape)
    div_term = 2.0 * jnp.mean(vmap(div_estimator, in_axes=(None, 0, 0))(score_fn.params, xv, eps))
    
    return norm_term + div_term

@partial(jit, static_argnums=(0,))
def train_score_step(train_state, particles_x, particles_v):
    """Single step of score network training"""
    def loss_fn(params):
        def score_fn(xv):
            return train_state.apply_fn({'params': params}, xv)
        return dsm_loss(score_fn, particles_x, particles_v)
    
    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    return train_state.apply_gradients(grads=grads), loss

def train_score_network(train_state, particles_x, particles_v, K):
    """Train score network for K steps"""
    for _ in range(K):
        train_state, loss = train_score_step(train_state, particles_x, particles_v)
    return train_state

# ============================
# Maxwell-Vlasov-Landau Evolution 
# ============================
@partial(jit, static_argnums=(2, 3, 4, 6, 7, 8))
def evolution_step(
    state,
    mesh_points,
    d_x,
    d_v,
    eta,
    dt,
    C,
    gamma,
    score_fn
):
    """
    Perform one step of Maxwell-Vlasov-Landau evolution with SBTM
    """
    particles_x, particles_v, E_mesh = state
    N = particles_x.shape[0]
    
    # 1. Interpolate electric field from mesh to particle positions
    def interp_E_single(x_p):
        distances = x_p - mesh_points
        weights = vmap(lambda d: psi_eta(d, eta))(distances)
        weights = weights / jnp.sum(weights)
        return jnp.sum(weights[:, None] * E_mesh, axis=0)
    
    E_particles = vmap(interp_E_single)(particles_x)
    
    # 2. Compute collision term U^{N,η,n}
    def compute_U_single(p):
        x_p, v_p = particles_x[p], particles_v[p]
        s_p = score_fn(jnp.concatenate([x_p, v_p]))
        
        def collision_term_q(q):
            x_q, v_q = particles_x[q], particles_v[q]
            s_q = score_fn(jnp.concatenate([x_q, v_q]))
            
            kernel = psi_eta(x_p - x_q, eta)
            A_matrix = compute_A_matrix(v_p - v_q, C, gamma)
            return kernel * jnp.dot(A_matrix, s_p - s_q)
        
        collisions = vmap(collision_term_q)(jnp.arange(N))
        return jnp.sum(collisions, axis=0) / N
    
    collision_terms = vmap(compute_U_single)(jnp.arange(N))
    
    # 3. Update velocities
    new_particles_v = particles_v + dt * (E_particles - collision_terms)
    
    # 4. Update positions
    velocity_spatial = vmap(lambda v: project_velocity(v, d_x, d_v))(new_particles_v)
    new_particles_x = particles_x + dt * velocity_spatial
    
    # Apply periodic boundary conditions to positions
    # Assuming domain is [0, L] in each dimension
    L = mesh_points[-1] - mesh_points[0] + (mesh_points[1] - mesh_points[0])
    new_particles_x = new_particles_x % L
    
    # 5. Update electric field on mesh
    def update_E_at_point(x_h, E_h):
        def contribution_from_particle(q):
            return psi_eta(x_h - particles_x[q], eta) * particles_v[q]
        
        particle_contributions = vmap(contribution_from_particle)(jnp.arange(N))
        return E_h - dt * jnp.sum(particle_contributions, axis=0) / N
    
    new_E_mesh = vmap(update_E_at_point)(mesh_points, E_mesh)
    
    return (new_particles_x, new_particles_v, new_E_mesh)

# ============================
# Initialization
# ============================
def initialize_particles_from_distribution(key, N, f0, domain_size):
    """
    Sample particles from initial distribution f0
    """
    # This is a placeholder. In practice, you'd need to implement proper sampling
    # from your specific f0 distribution
    
    # For Landau damping example from the paper
    x_key, v_key = random.split(key)
    
    # Uniform positions in [0, L]
    particles_x = random.uniform(x_key, (N, 1)) * domain_size
    
    # Velocities from Maxwellian with perturbation
    particles_v1 = random.normal(v_key, (N, 1))
    v_key, _ = random.split(v_key)
    particles_v2 = random.normal(v_key, (N, 1))
    particles_v = jnp.concatenate([particles_v1, particles_v2], axis=1)
    
    # Apply perturbation based on positions (for Landau damping)
    # This is simplified; you'll need to adapt to your specific f0
    alpha = 0.1
    k = 0.5
    perturbation = 1 + alpha * jnp.cos(k * particles_x[:, 0])
    
    # Rejection sampling based on perturbation
    u_key, _ = random.split(v_key)
    u = random.uniform(u_key, (N, 1))
    keep = u <= (perturbation[:, None] / (1 + alpha))
    
    # In practice, you'd resample rejected particles
    # Here we're just returning all particles for simplicity
    
    return particles_x, particles_v

def initialize_system(
    key,
    N,
    f0,
    mesh_size,
    domain_size,
    d_x,
    d_v,
    rho_ion,
    score_net_features,
    K0
):
    """
    Initialize the particle system, electric field, and score network
    """
    # Sample particles from f0
    particles_x, particles_v = initialize_particles_from_distribution(key, N, f0, domain_size)
    
    # Discretize domain
    dx = domain_size / mesh_size
    eta = jnp.ones(d_x) * dx
    mesh_points = jnp.linspace(dx/2, domain_size - dx/2, mesh_size).reshape(-1, 1)
    
    # Compute initial charge density on mesh
    def compute_rho_at_point(x_h):
        def contribution(p):
            return psi_eta(x_h - particles_x[p], eta)
        return jnp.sum(vmap(contribution)(jnp.arange(N))) / N
    
    rho = vmap(compute_rho_at_point)(mesh_points)
    
    # Set up and solve Poisson equation
    if d_x == 1:
        laplacian = build_1d_laplacian(mesh_size, dx)
    else:
        # Would need implementation for higher dimensions
        raise NotImplementedError("Only 1D spatial domain is currently supported")
    
    phi = solve_poisson_equation(rho, rho_ion, laplacian)
    E0 = compute_electric_field_from_phi(phi, dx, d_x)
    
    # Initialize score network
    model = ScoreNetwork(features=score_net_features)
    input_shape = (d_x + d_v,)
    params_key, train_key = random.split(key)
    variables = model.init(params_key, jnp.ones(input_shape))
    
    # Set up optimizer
    tx = optax.adam(learning_rate=1e-3)
    train_state = optax.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )
    
    # Train initial score network
    if K0 > 0:
        # Here we'd implement training on the true score of f0
        # For demonstration, we're just using regular score matching
        train_state = train_score_network(train_state, particles_x, particles_v, K0)
    
    return (particles_x, particles_v, E0), mesh_points, train_state

# ============================
# Main Simulation Loop
# ============================
def run_simulation(
    key,
    N,
    f0,
    mesh_size,
    domain_size,
    d_x,
    d_v,
    T,
    dt,
    C,
    gamma,
    rho_ion,
    score_net_features,
    K0,
    K
):
    """
    Run the full Vlasov-Maxwell-Landau simulation with SBTM
    """
    # Initialize system
    state, mesh_points, train_state = initialize_system(
        key, N, f0, mesh_size, domain_size, d_x, d_v, rho_ion, score_net_features, K0
    )
    
    # Storage for diagnostics
    electric_energy = []
    
    # Main time-stepping loop
    for t in range(T):
        # Record diagnostics
        particles_x, particles_v, E_mesh = state
        E_energy = jnp.sum(jnp.sum(E_mesh**2, axis=1)) * (domain_size / mesh_size)
        electric_energy.append(E_energy)
        
        # Define score function for current iteration
        def score_fn(xv):
            return train_state.apply_fn({'params': train_state.params}, xv)
        
        # Evolve system one step
        state = evolution_step(
            state, mesh_points, d_x, d_v, 
            jnp.ones(d_x) * (domain_size / mesh_size),
            dt, C, gamma, score_fn
        )
        
        # Train score network on new particle distribution
        particles_x, particles_v, _ = state
        train_state = train_score_network(train_state, particles_x, particles_v, K)
        
        # Optional: Print progress
        if t % 10 == 0:
            print(f"Step {t}/{T}, Electric field energy: {E_energy}")
    
    return state, train_state, jnp.array(electric_energy)

# ============================
# Landau Damping Example
# ============================
def landau_damping_example():
    """
    Run the Landau damping example from the paper
    """
    # Parameters from the paper
    d_x = 1
    d_v = 2
    gamma = -d_v  # Coulombian collisions
    alpha = 0.1
    k = 0.5
    domain_size = 2 * jnp.pi / k
    T = 10
    dt = 1/50
    steps = int(T / dt)
    
    # Define initial distribution (for reference)
    def f0(x, v):
        return (1 + alpha * jnp.cos(k * x)) / (2 * jnp.pi) * jnp.exp(-(v[0]**2 + v[1]**2) / 2)
    
    # Particle and mesh parameters
    N_x = 128
    N_v = 32
    N_c = 8
    N = N_x * (N_v**2) * N_c  # 2^20 particles
    mesh_size = N_x
    
    # Network parameters
    score_net_features = [64, 64, 64, d_v]  # Output dimension matches velocity space
    K0 = 100  # Initial training steps
    K = 10    # Training steps per iteration
    
    # Collision strengths to test
    collision_strengths = [0.0, 0.01, 0.1, 1.0]
    
    # Run simulations for different collision strengths
    results = {}
    key = random.PRNGKey(0)
    
    for C in collision_strengths:
        print(f"Running simulation with collision strength C = {C}")
        key, subkey = random.split(key)
        
        final_state, final_score, electric_energy = run_simulation(
            subkey, N, f0, mesh_size, domain_size, d_x, d_v, steps, dt, C, gamma,
            0.0,  # rho_ion (no background ions)
            score_net_features, K0, K
        )
        
        results[C] = {
            'final_state': final_state,
            'electric_energy': electric_energy
        }
    
    return results

if __name__ == "__main__":
    results = landau_damping_example()
    
    # Plot results
    # This would typically be done using matplotlib
    print("Simulation complete. Final electric field energy values:")
    for C, result in results.items():
        print(f"C = {C}: {float(result['electric_energy'][-1])}")