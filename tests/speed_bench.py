#%%
import jax
import jax.numpy as jnp
import jax.random as jrandom

from functools import partial
import time
import jax.lax as lax
from flax import nnx
import optax
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

#%%
@jax.jit
def update_electric_field(E, cells, x, v, eta, dt, box_length):
    M = cells.size

    idx_f = x / eta - 0.5
    i0    = jnp.floor(idx_f).astype(jnp.int32) % M
    f     = idx_f - jnp.floor(idx_f)
    i1    = (i0 + 1) % M
    w0, w1 = 1.0 - f, f

    J = (
        jnp.zeros(M)
          .at[i0].add(w0 * v[:, 0])
          .at[i1].add(w1 * v[:, 0])
        / (x.size * eta)
    )

    return (E - dt * box_length * J).astype(E.dtype)

@jax.jit
def evaluate_field_at_particles(x, cells, E, eta, box_length):
    """
    eta * Σ_j ψ(x_i − cell_j) E_j   (linear-hat kernel, periodic)
    Now O(N): two-point linear interpolation of E instead of a full sum.
    """
    M      = cells.size
    idx_f  = x / eta - 0.5
    i0     = jnp.floor(idx_f).astype(jnp.int32) % M
    f      = idx_f - jnp.floor(idx_f)
    i1     = (i0 + 1) % M
    return (1.0 - f) * E[i0] + f * E[i1]

@jax.jit
def evaluate_charge_density(x, cells, eta, box_length, qe=1.0):
    """
    ρ_j = qe * box_length * ⟨ψ(x − cell_j)⟩   with ψ the same hat kernel.
    O(N) scatter-add instead of vmap over cells.
    """
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
#%%
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

# --- collision kernel --------------------------------------------------------
@jax.jit
def A(z, C, gamma):
    """A(z) = C |z|^gamma (|z|² I_d − z⊗z)."""
    z_norm  = jnp.linalg.norm(z) + 1e-10        # avoid divide-by-zero
    factor  = C * z_norm ** gamma
    return factor * (jnp.eye(z.shape[0]) * z_norm**2 - jnp.outer(z, z))

# --- fast ψ-hat collision ----------------------------------------------------
@partial(jax.jit, static_argnums=6)    # ← num_cells is concrete Python int
def collision_hat_local(x, v, s, eta, C, gamma,
                        num_cells,          # int(box_length / eta)
                        box_length):
    """
    Q_i = (1/N) Σ_{|x_i−x_j|≤η} ψ(x_i−x_j) · A(v_i−v_j)(s_i−s_j)
          with the linear-hat kernel ψ of width η, periodic on [0,L].

    Complexity O(N η/L)  (exact for the hat kernel).
    """
    N, d = v.shape
    M    = num_cells                        # concrete Python int

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
    return Q_sorted[rev]                  # shape (N, d)

#%%

seed = 42

# set physical constants
alpha = 0.1  # Perturbation strength
k = 0.5      # Wave number
dx = 1       # Position dimension
dv = 2       # Velocity dimension

num_particles = 1_000_0

# Create a mesh
box_length = 2 * jnp.pi / k
num_cells = 128
eta = box_length / num_cells
cells = (jnp.arange(num_cells) + 0.5) * eta

# sample initial velocity
key_v, key_x = jrandom.split(jrandom.PRNGKey(seed), 2)
v0 = jrandom.multivariate_normal(key_v, jnp.zeros(dv), jnp.eye(dv), shape=(num_particles,)).reshape((num_particles, dv))

# Sample initial positions with rejection sampling
x0 = jrandom.uniform(key_x, (num_particles,), minval=0, maxval=box_length)

# Compute initial electric field
rho = evaluate_charge_density(x0, cells, eta, box_length)
E0 = jnp.cumsum(rho - jnp.mean(rho)) * eta

s = jrandom.normal(jrandom.PRNGKey(0), (num_particles, dv))

# %%
"Benchmark collision operators"

C = 0.1
gamma = -dv

# Time collision_hat_local
collision_hat_local(x0, v0, s, eta, C, gamma, num_cells, box_length) # can even handle 10^6 particles, takes ~9 seconds
start1 = time.time()
c1 = collision_hat_local(x0, v0, s, eta, C, gamma, num_cells, box_length)
jax.block_until_ready(c1)
end1 = time.time()
print(f"collision_hat_local time: {end1 - start1:.4f} s")

# Time collision
collision(x0, v0, s, eta, C, gamma, box_length)
start2 = time.time()
c2 = collision(x0, v0, s, eta, C, gamma, box_length) # O(N^2), so N must be <10_000
jax.block_until_ready(c2)
end2 = time.time()
print(f"collision (naive) time: {end2 - start2:.4f} s")

print(jnp.max(jnp.abs(c1 - c2)))  # should be small

# %%
@jax.jit
def evaluate_field_at_particles_old(x, cells, E, eta, box_length):
    """Evaluate electric field at particle positions."""
    return jax.vmap(lambda x_i: eta * jnp.sum(psi(x_i - cells, eta, box_length) * E, axis=0))(x)

@jax.jit
def evaluate_charge_density_old(x, cells, eta, box_length, qe=1):
    rho = qe * box_length * jax.vmap(lambda cell: jnp.mean(psi(x - cell, eta, box_length)))(cells)
    return rho

@jax.jit
def update_electric_field_old(E, cells, x, v, eta, dt, box_length):
    """Update electric field on the mesh."""
    kernel_values = psi(cells[:, None] - x[None, :], eta, box_length)
    return E - dt * box_length * jnp.mean(kernel_values * v[:,0], axis=1)

# %%
"Benchmark electric field evaluation and update"

num_particles = 1_000_000
key_v, key_x = jrandom.split(jrandom.PRNGKey(seed), 2)
v0 = jrandom.multivariate_normal(key_v, jnp.zeros(dv), jnp.eye(dv), shape=(num_particles,)).reshape((num_particles, dv))
x0 = jrandom.uniform(key_x, (num_particles,), minval=0, maxval=box_length)


E1 = evaluate_field_at_particles_old(x0, cells, E0, eta, box_length)
E2 = evaluate_field_at_particles(x0, cells, E0, eta, box_length)
print(jnp.max(jnp.abs(E1 - E2)))
rho1 = evaluate_charge_density_old(x0, cells, eta, box_length)
rho2 = evaluate_charge_density(x0, cells, eta, box_length)
print(jnp.max(jnp.abs(rho1 - rho2)))

E1 = update_electric_field_old(E0, cells, x0, v0, eta, 0.01, box_length)
E2 = update_electric_field(E0, cells, x0, v0, eta, 0.01, box_length)
print(jnp.max(jnp.abs(E1 - E2)))

start1 = time.time()
update_electric_field_old(E0, cells, x0, v0, eta, 0.01, box_length).block_until_ready()
end1 = time.time()
print(f"update_electric_field_old time: {end1 - start1:.4f} s")
start2 = time.time()
update_electric_field(E0, cells, x0, v0, eta, 0.01, box_length).block_until_ready()
end2 = time.time()
print(f"update_electric_field time: {end2 - start2:.4f} s")

start1 = time.time()
E1 = evaluate_field_at_particles_old(x0, cells, E0, eta, box_length).block_until_ready()
end1 = time.time()
print(f"evaluate_field_at_particles_old time: {end1 - start1:.4f} s")
start2 = time.time()
E2 = evaluate_field_at_particles(x0, cells, E0, eta, box_length).block_until_ready()
end2 = time.time()
print(f"evaluate_field_at_particles time: {end2 - start2:.4f} s")

start1 = time.time()
rho1 = evaluate_charge_density_old(x0, cells, eta, box_length).block_until_ready()
end1 = time.time()
print(f"evaluate_charge_density_old time: {end1 - start1:.4f} s")
start2 = time.time()
rho2 = evaluate_charge_density(x0, cells, eta, box_length).block_until_ready()
end2 = time.time()
print(f"evaluate_charge_density time: {end2 - start2:.4f} s")

# %%
"CPU vs GPU speed comparison"


num_particles = 10_000
key_v, key_x = jrandom.split(jrandom.PRNGKey(seed), 2)
v0 = jrandom.multivariate_normal(key_v, jnp.zeros(dv), jnp.eye(dv), shape=(num_particles,)).reshape((num_particles, dv))
x0 = jrandom.uniform(key_x, (num_particles,), minval=0, maxval=box_length)

# Ensure arrays are on GPU
x0_gpu = x0
v0_gpu = v0
s_gpu = s
E0_gpu = E0

# Move arrays to CPU
x0_cpu = jax.device_put(x0, device=jax.devices("cpu")[0])
v0_cpu = jax.device_put(v0, device=jax.devices("cpu")[0])
s_cpu = jax.device_put(s, device=jax.devices("cpu")[0])
E0_cpu = jax.device_put(E0, device=jax.devices("cpu")[0])

# --- collision_hat_local benchmark ---
print("Benchmarking collision_hat_local (GPU)...")
collision_hat_local(x0_gpu, v0_gpu, s_gpu, eta, C, gamma, num_cells, box_length)
start = time.time()
c_gpu = collision_hat_local(x0_gpu, v0_gpu, s_gpu, eta, C, gamma, num_cells, box_length)
jax.block_until_ready(c_gpu)
end = time.time()
print(f"collision_hat_local (GPU) time: {end - start:.4f} s")

print("Benchmarking collision_hat_local (CPU)...")
collision_hat_local(x0_cpu, v0_cpu, s_cpu, eta, C, gamma, num_cells, box_length)
start = time.time()
c_cpu = collision_hat_local(x0_cpu, v0_cpu, s_cpu, eta, C, gamma, num_cells, box_length)
jax.block_until_ready(c_cpu)
end = time.time()
print(f"collision_hat_local (CPU) time: {end - start:.4f} s") # 0.3 s

print("Max abs diff (GPU vs CPU):", jnp.max(jnp.abs(c_gpu - c_cpu)))


num_particles = 1_000_000
key_v, key_x = jrandom.split(jrandom.PRNGKey(seed), 2)
v0 = jrandom.multivariate_normal(key_v, jnp.zeros(dv), jnp.eye(dv), shape=(num_particles,)).reshape((num_particles, dv))
x0 = jrandom.uniform(key_x, (num_particles,), minval=0, maxval=box_length)

# Ensure arrays are on GPU
x0_gpu = x0
v0_gpu = v0
s_gpu = s
E0_gpu = E0

# Move arrays to CPU
x0_cpu = jax.device_put(x0, device=jax.devices("cpu")[0])
v0_cpu = jax.device_put(v0, device=jax.devices("cpu")[0])
s_cpu = jax.device_put(s, device=jax.devices("cpu")[0])
E0_cpu = jax.device_put(E0, device=jax.devices("cpu")[0])


# --- evaluate_field_at_particles benchmark ---
print("Benchmarking evaluate_field_at_particles (GPU)...")
evaluate_field_at_particles(x0_gpu, cells, E0_gpu, eta, box_length)
start = time.time()
E_gpu = evaluate_field_at_particles(x0_gpu, cells, E0_gpu, eta, box_length)
jax.block_until_ready(E_gpu)
end = time.time()
print(f"evaluate_field_at_particles (GPU) time: {end - start:.4f} s")

print("Benchmarking evaluate_field_at_particles (CPU)...")
evaluate_field_at_particles(x0_cpu, cells, E0_cpu, eta, box_length)
start = time.time()
E_cpu = evaluate_field_at_particles(x0_cpu, cells, E0_cpu, eta, box_length)
jax.block_until_ready(E_cpu)
end = time.time()
print(f"evaluate_field_at_particles (CPU) time: {end - start:.4f} s")

print("Max abs diff (GPU vs CPU):", jnp.max(jnp.abs(E_gpu - E_cpu)))

# --- evaluate_charge_density benchmark ---
print("Benchmarking evaluate_charge_density (GPU)...")
evaluate_charge_density(x0_gpu, cells, eta, box_length)
start = time.time()
rho_gpu = evaluate_charge_density(x0_gpu, cells, eta, box_length)
jax.block_until_ready(rho_gpu)
end = time.time()
print(f"evaluate_charge_density (GPU) time: {end - start:.4f} s")

print("Benchmarking evaluate_charge_density (CPU)...")
evaluate_charge_density(x0_cpu, cells, eta, box_length)
start = time.time()
rho_cpu = evaluate_charge_density(x0_cpu, cells, eta, box_length)
jax.block_until_ready(rho_cpu)
end = time.time()
print(f"evaluate_charge_density (CPU) time: {end - start:.4f} s")

print("Max abs diff (GPU vs CPU):", jnp.max(jnp.abs(rho_gpu - rho_cpu)))

# --- update_electric_field benchmark ---
print("Benchmarking update_electric_field (GPU)...")
update_electric_field(E0_gpu, cells, x0_gpu, v0_gpu, eta, 0.01, box_length)
start = time.time()
E_update_gpu = update_electric_field(E0_gpu, cells, x0_gpu, v0_gpu, eta, 0.01, box_length)
jax.block_until_ready(E_update_gpu)
end = time.time()
print(f"update_electric_field (GPU) time: {end - start:.4f} s")

print("Benchmarking update_electric_field (CPU)...")
update_electric_field(E0_cpu, cells, x0_cpu, v0_cpu, eta, 0.01, box_length)
start = time.time()
E_update_cpu = update_electric_field(E0_cpu, cells, x0_cpu, v0_cpu, eta, 0.01, box_length)
jax.block_until_ready(E_update_cpu)
end = time.time()
print(f"update_electric_field (CPU) time: {end - start:.4f} s")

print("Max abs diff (GPU vs CPU):", jnp.max(jnp.abs(E_update_gpu - E_update_cpu)))

# %%
def divergence_wrt_v(f, mode: str, num_noise: int = 1):
    assert mode in ['forward', 'reverse', 'approximate_gaussian', 'approximate_rademacher', 'denoised'], "Invalid mode"
    
    # Create a wrapper that treats only v as the variable for differentiation
    def f_wrapper(v, x):
        return f(x, v)
    
    if mode == 'forward':
        @jax.jit
        def div(x, v):
            return jnp.trace(jax.jacfwd(f_wrapper, argnums=0)(v, x))
        return div
        
    if mode == 'reverse':
        @jax.jit
        def div(x, v):
            return jnp.trace(jax.jacrev(f_wrapper, argnums=0)(v, x))
        return div
        
    if mode == 'denoised':
        alpha = jnp.float32(0.1)
        @jax.jit
        def div(x, v, key):
            def denoise(key):
                epsilon = jax.random.normal(key, v.shape, dtype=v.dtype)
                return jnp.sum(
                    (f(x, v + alpha * epsilon) - f(x, v - alpha * epsilon)) * epsilon
                ) / (2*alpha)
            return jax.vmap(denoise)(jax.random.split(key, num_noise)).mean()
        return div
    else:
        @jax.jit
        def div(x, v, key):
            def vJv(key):
                # Define a partial function that fixes x
                fixed_x_f = lambda v_: f(x, v_)
                # Get vector-Jacobian product function
                _, vjp_fun = jax.vjp(fixed_x_f, v)
                # Generate random vector
                rand_gen = jax.random.normal if mode == 'approximate_gaussian' else jax.random.rademacher
                epsilon = rand_gen(key, v.shape, dtype=v.dtype)
                # Compute vᵀ(∂f/∂v)ᵀv = vᵀJ_v[f]ᵀv
                return jnp.sum(vjp_fun(epsilon)[0] * epsilon)
            return jax.vmap(vJv)(jax.random.split(key, num_noise)).mean()
        return div

# %%
from flax import nnx
import optax
from src.score_model import MLPScoreModel
import time

dx = 1
dv = 3
seed = 42
box_length = 1
model = MLPScoreModel(dx, dv, hidden_dims=(64,))
num_particles = 1_000_000
key_v, key_x = jrandom.split(jrandom.PRNGKey(seed), 2)
v = jrandom.multivariate_normal(key_v, jnp.zeros(dv), jnp.eye(dv), shape=(num_particles,))
x = jrandom.uniform(key_x, (num_particles,1), minval=0, maxval=box_length)

for div_mode in ['forward', 'reverse']:
    print(f"Testing divergence mode: {div_mode}")
    div_fn = divergence_wrt_v(model, div_mode, 1)

    div = div_fn(x[0],v[0])
    print("Divergence:", div)

    @jax.jit
    def compute_mean_div(x, v):
        return jnp.mean(jax.vmap(div_fn)(x, v))

    _ = compute_mean_div(x, v)

    # ── Timed execution-only run ──
    mean_div = compute_mean_div(x, v)
    print("Mean divergence:", mean_div)

    times = []
    for _ in range(20):
        start = time.time()
        out = compute_mean_div(x, v)
        jax.block_until_ready(out)
        times.append(time.time() - start)

    print(f"Avg execution time: {sum(times)/len(times):.4f} s")

for div_mode in ['approximate_gaussian', 'approximate_rademacher', 'denoised']:
    print(f"Testing divergence mode: {div_mode}")
    div_fn = divergence_wrt_v(model, div_mode, 1)

    key = jax.random.PRNGKey(0)
    div = div_fn(x[0], v[0], key=key)
    print("Divergence:", div)

    @jax.jit
    def compute_mean_div(x, v, key):
        keys = jax.random.split(key, x.shape[0])
        return jnp.mean(jax.vmap(div_fn)(x, v, keys))

    _ = compute_mean_div(x, v, key)

    # ── Timed execution-only run ──
    mean_div = compute_mean_div(x, v, key)
    print("Mean divergence:", mean_div)

    times = []
    for _ in range(20):
        start = time.time()
        out = compute_mean_div(x, v, key)
        jax.block_until_ready(out)
        times.append(time.time() - start)

    print(f"Avg execution time: {sum(times)/len(times):.4f} s")

#%%
@nnx.jit(static_argnames=['div_mode', 'n_samples'])
def implicit_score_matching_loss(s, x_batch, v_batch, key=None, div_mode='reverse', n_samples=1):
    # Get the appropriate divergence function
    div_fn = divergence_wrt_v(s, div_mode, n_samples)
    
    def compute_loss(x, v, key=None):
        # Compute squared norm of score
        score = s(x, v)
        squared_norm = jnp.sum(jnp.square(score))
        
        # Compute divergence based on mode
        if div_mode in ['approximate_gaussian', 'approximate_rademacher', 'denoised']:
            assert key is not None, "For stochastic divergence estimation, key must be provided"
            div = div_fn(x, v, key) if key is not None else div_fn(x, v)
        else:
            div = div_fn(x, v)
            
        return squared_norm + 2 * div
    
    if div_mode in ['forward', 'reverse']:
        # For exact methods, we can directly vmap over the batch
        return jnp.mean(jax.vmap(compute_loss)(x_batch, v_batch))
    else:
        # For stochastic methods, we need to handle the random keys
        batch_size = x_batch.shape[0]
        keys = jax.random.split(key, batch_size)
        loss_fn = lambda x, v, k: compute_loss(x, v, k)
        return jnp.mean(jax.vmap(loss_fn)(x_batch, v_batch, keys))

def train_score_model(score_model, x_batch, v_batch, training_config):
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

#%%
from flax import nnx
import optax
from src.score_model import MLPScoreModel
import time

dx = 1
dv = 3
seed = 42
box_length = 1
model = MLPScoreModel(dx, dv, hidden_dims=(64,))
num_particles = 1_000_000
key_v, key_x = jrandom.split(jrandom.PRNGKey(seed), 2)
v = jrandom.multivariate_normal(key_v, jnp.zeros(dv), jnp.eye(dv), shape=(num_particles,))
x = jrandom.uniform(key_x, (num_particles,1), minval=0, maxval=box_length)

for div_mode in ['forward', 'reverse', 'approximate_gaussian', 'approximate_rademacher', 'denoised']:
    print(f"Testing divergence mode: {div_mode}")
    key = jax.random.PRNGKey(0)
    loss_val = implicit_score_matching_loss(model, x, v, key, div_mode=div_mode)
    print("Loss value:", loss_val)
    
    times = []
    for _ in range(50):
        start = time.time()
        out = implicit_score_matching_loss(model, x, v, key, div_mode=div_mode)
        jax.block_until_ready(out)
        times.append(time.time() - start)

    print(f"Avg execution time: {sum(times)/len(times):.4f} s")

#%%
# TODO: bench gradient descent
optimizer = optax.adamw(1e-3)
opt_step(model, optimizer, implicit_score_matching_loss, (x, v), key=None)