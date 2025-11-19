#%%
import math
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import time
from functools import partial
import os
import numpy as np

jax.config.update("jax_traceback_filtering", "off")
jax.config.update("jax_enable_x64", True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# runtime
def bench(fn, *args, repeat=3, name=None, **kwargs):
    # warmup (exclude compile time)
    out = fn(*args, **kwargs)
    jax.block_until_ready(out)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    print(f"[{name or fn.__name__}] {min(times)*1e3:.2f} ms (min of {repeat})")
    return out

# memory, flops
def cost(fn, *args, name=None, **kwargs):
    comp = fn.lower(*args, **kwargs).compile()
    ca = comp.cost_analysis()
    ba = ca.get("bytes accessed", 0)
    fl = ca.get("flops", 0)
    print(f"[{name or fn.__name__}] bytes_accessed={ba/1e6:.1f} MB, flops={fl/1e9:.2f} GFLOP")
    return comp


#%%
@jax.jit
def psi(x, eta, box_length):
    x = (x + 0.5 * box_length) % box_length - 0.5 * box_length   # centered_mod
    return jnp.maximum(0.0, 1.0 - jnp.abs(x / eta)) / eta

@jax.jit
def A(z, C, gamma):
    z_norm  = jnp.linalg.norm(z) + 1e-10
    factor  = C * z_norm ** gamma
    return factor * (jnp.eye(z.shape[-1]) * z_norm**2 - jnp.outer(z, z))

@partial(jax.jit, static_argnames='num_cells')
def collision(x, v, s, eta, C, gamma, box_length, num_cells):
    """
    Q_i = (L/N) Σ_{|x_i−x_j|≤η} ψ(x_i−x_j) · A(v_i−v_j)(s_i−s_j)
          with the linear-hat kernel ψ of width η, periodic on [0,L].

    Complexity O(N η/L)  (exact for the hat kernel).
    """
    x = x[:, 0]
    N, d      = v.shape
    M         = num_cells
    w_particle = box_length / N        # (= L/N)
    # w_particle = 1 / N        # (= L/N)

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
                ψ  = psi(xi - xj, eta, box_length)
                dv = vi - vj
                ds = si - sj
                inner_acc += ψ * jnp.dot(A(dv, C, gamma), ds)
                return inner_acc

            acc = lax.fori_loop(start, end, loop_over_particles, acc)
            return acc

        # neighbour cells that overlap the hat (periodic)
        for c in ((cell - 1) % M, cell, (cell + 1) % M):
            Q_i = loop_over_cell(c, Q_i)

        return Q_i

    Q_sorted = jax.vmap(Q_single)(jnp.arange(N))

    # ---- unsort back to original particle order ----------------------------
    rev = jnp.empty_like(order)
    rev = rev.at[order].set(jnp.arange(N))
    return w_particle * Q_sorted[rev]

def A_apply(dv, ds, C, gamma, eps=1e-10):
    r2   = jnp.dot(dv, dv) + eps          # ‖dv‖²
    r_g  = r2 ** (gamma / 2)              # ‖dv‖^γ
    dvds = jnp.dot(dv, ds)                # dv·ds
    return C * r_g * (r2 * ds - dvds * dv)

@partial(jax.jit, static_argnames='num_cells')
def collision_2(x, v, s, eta, C, gamma, box_length, num_cells):
    """
    Q_i = (L/N) Σ_{|x_i−x_j|≤η} ψ(x_i−x_j) · A(v_i−v_j)(s_i−s_j)
          with the linear-hat kernel ψ of width η, periodic on [0,L].

    Complexity O(N η/L)  (exact for the hat kernel).
    """
    x = x[:, 0]
    N, d      = v.shape
    M         = num_cells
    w_particle = box_length / N        # (= L/N)
    # w_particle = 1 / N        # (= L/N)

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
                ψ  = psi(xi - xj, eta, box_length)
                dv = vi - vj
                ds = si - sj
                inner_acc += ψ * A_apply(dv, ds, C, gamma)
                return inner_acc

            acc = lax.fori_loop(start, end, loop_over_particles, acc)
            return acc

        # neighbour cells that overlap the hat (periodic)
        for c in ((cell - 1) % M, cell, (cell + 1) % M):
            Q_i = loop_over_cell(c, Q_i)

        return Q_i

    Q_sorted = jax.vmap(Q_single)(jnp.arange(N))

    # ---- unsort back to original particle order ----------------------------
    rev = jnp.empty_like(order)
    rev = rev.at[order].set(jnp.arange(N))
    return w_particle * Q_sorted[rev]

# THERE IS NO BEST VERSION
# on l40s, with fp32, n=4e5, M=100, dv=2, takes ~1.2s and 250Mb memory
# on l40s, with fp32, n=1e6, M=100, dv=2, takes ~6.4s and 640Mb memory
# on rtx6k, with fp64, n=4e5, M=100, dv=2, takes ~5.1s and 550Mb memory
# on rtx6k, with fp64, n=1e6, M=100, dv=2, takes ~31s and 1.4Gb memory
@partial(jax.jit, static_argnames=['num_cells'])
def collision_3(x, v, s, eta, gamma, num_cells, box_length, w):
    """ 
    Q_i = w Σ_{|x_i−x_j|≤η} ψ(x_i−x_j) A(v_i−v_j)(s_i−s_j) with the linear-hat kernel ψ of width eta, periodic on [0,L]. 
    Complexity O(N η/L). 
    """
    def A_apply(dv, ds, gamma, eps=1e-14):
        v2 = jnp.sum(dv * dv, axis=-1, keepdims=True) + eps
        vg = v2 ** (gamma / 2)
        dvds = jnp.sum(dv * ds, axis=-1, keepdims=True)
        return vg * (v2 * ds - dvds * dv)
    if x.ndim == 2:
        x = x[:, 0]
    N, d   = v.shape
    M      = num_cells

    # bin + sort
    cell = (jnp.floor(x / eta).astype(jnp.int32)) % M
    order = jnp.argsort(cell)
    x, v, s, cell = x[order], v[order], s[order], cell[order]

    counts = jnp.bincount(cell, length=M)
    starts = jnp.cumsum(jnp.concatenate([jnp.array([0]), counts[:-1]]))
    
    def centered_mod(x, L):
        "centered_mod(x, L) in [-L/2, L/2]"
        return (x + L/2) % L - L/2

    def psi(x, eta, box_length):
        "psi_eta(x) = max(0, 1-|x|/eta) / eta."
        x = centered_mod(x, box_length)
        kernel = jnp.maximum(0.0, 1.0 - jnp.abs(x / eta))
        return kernel / eta

    def Q_single(i):
        xi, vi, si = x[i], v[i], s[i]
        ci = cell[i]
        acc = jnp.zeros(d)

        for c in ((ci - 1) % M, ci, (ci + 1) % M):
            start = starts[c]
            end   = start + counts[c]

            def add_j(j, accj):
                ψ  = psi(xi - x[j], eta, box_length)
                dv = vi - v[j]
                ds = si - s[j]
                return accj + ψ * A_apply(dv, ds, gamma)

            acc = lax.fori_loop(start, end, add_j, acc)
        return acc

    Q_sorted = jax.vmap(Q_single)(jnp.arange(N))

    rev = jnp.empty_like(order).at[order].set(jnp.arange(N))
    return w * Q_sorted[rev]

# --- Gemini's attempt: collision_4 with dynamic window size ---
@jax.jit
def A_apply_new(dv, ds, gamma, eps=1e-14):
    """Applies the collision operator A to velocity and signal differences."""
    v2 = jnp.sum(dv * dv, axis=-1, keepdims=True) + eps
    # Use exp/log for stability with variable gamma, or simple power if gamma is int
    vg = jnp.exp(0.5 * gamma * jnp.log(v2)) 
    dvds = jnp.sum(dv * ds, axis=-1, keepdims=True)
    return vg * (v2 * ds - dvds * dv)

# We mark 'window_size' as static so JAX can unroll the loops efficiently.
@partial(jax.jit, static_argnames=['window_size'])
def collision_kernel(x, v, s, eta, gamma, box_length, w, window_size):
    if x.ndim == 2:
        x = x[:, 0]
    
    N, d = v.shape
    
    # A. Sort particles spatially (1D)
    order = jnp.argsort(x)
    x_sorted = x[order]
    v_sorted = v[order]
    s_sorted = s[order]
    
    # B. Create Neighbor Indices (The Sliding Window)
    # Creates a matrix of shape (N, 2*window_size + 1)
    offsets = jnp.arange(-window_size, window_size + 1)
    neighbor_indices = (jnp.arange(N)[:, None] + offsets[None, :]) % N
    
    # C. Gather Neighbor Data
    x_neighbors = x_sorted[neighbor_indices] # (N, Window, 1)
    v_neighbors = v_sorted[neighbor_indices] # (N, Window, d)
    s_neighbors = s_sorted[neighbor_indices] # (N, Window, d)
    
    # D. Compute Distances & Periodic Wrapping
    dx = x_sorted[:, None] - x_neighbors
    # Minimum Image Convention for 1D periodic boundary
    dx = (dx + box_length / 2) % box_length - box_length / 2
    dist = jnp.abs(dx)

    # E. Masking
    # 1. Ignore neighbors outside interaction radius 'eta'
    # 2. Ignore self-interaction (dist > 0)
    mask = (dist <= eta) & (dist > 0.0)
    
    # F. Linear Hat Kernel
    psi = jnp.maximum(0.0, 1.0 - dist / eta) / eta
    psi = psi * mask # Zero out invalid interactions
    
    # G. Physics Interaction (Vectorized)
    # Broadcast (N, 1, d) against (N, Window, d)
    dv_vec = v_sorted[:, None, :] - v_neighbors
    ds_vec = s_sorted[:, None, :] - s_neighbors
    
    interaction = A_apply_new(dv_vec, ds_vec, gamma) # (N, Window, d)
    
    # Sum over the window dimension (axis 1)
    # We broadcast psi (N, Window) to (N, Window, 1)
    Q_sorted = jnp.sum(interaction * psi[:, :, None], axis=1)
    
    # H. Unsort (Scatter back to original order)
    rev = jnp.empty_like(order).at[order].set(jnp.arange(N))
    return w * Q_sorted[rev]

def compute_window_params(N, eta, box_length, bucket_size=100, safety_factor=1.2):
    """
    Calculates the required window size based on density and buckets it.
    
    Args:
        N: Number of particles
        eta: Interaction radius
        box_length: Domain size
        bucket_size: Round up to multiples of this (prevents frequent recompilation)
        safety_factor: Multiplier to account for particle clustering (non-uniformity)
    """
    # Average density of particles
    density = N / box_length
    
    # Expected neighbors in range [-eta, +eta] (so 2 * eta)
    # We only need 'window_size' which is the radius (neighbors on ONE side)
    avg_neighbors_radius = density * eta
    
    # Add safety margin for clustering
    target_size = avg_neighbors_radius * safety_factor
    
    # Round up to nearest bucket_size (e.g., 100)
    # Formula: ceil(x / 100) * 100
    window_size = math.ceil(target_size / bucket_size) * bucket_size
    
    # Clamp: Window cannot be larger than N//2 (wrapping around the world)
    window_size = min(int(window_size), N // 2)
    
    # Ensure at least 1 to avoid shape errors
    return max(window_size, 1)
    
def collision_4(x, v, s, eta, gamma, box_length, w):
    """
    Driver function that calculates window size and dispatches to JIT kernel.
    """
    N = v.shape[0]
    
    # 1. Pre-compute the window size (Python int)
    # This is fast and happens outside JAX.
    win_size = compute_window_params(N, eta, box_length)
    
    # 2. Call the JIT-compiled kernel
    # JAX will compile a version for 'win_size=100'.
    # If density increases and win_size becomes 200, JAX compiles a new version.
    return collision_kernel(x, v, s, eta, gamma, box_length, w, window_size=win_size)

# --- Gemini's second attempt: collision_5 with smaller memory footprint ---
@partial(jax.jit, static_argnames=['window_size'])
def collision_rolling(x, v, s, eta, gamma, box_length, w, window_size):
    """
    O(N) Memory implementation using Vectorized Rolling.
    """
    if x.ndim == 2:
        x = x[:, 0]
    
    N, d = v.shape

    # 1. Sort (Essential for windowing)
    order = jnp.argsort(x)
    x_sorted = x[order]
    v_sorted = v[order]
    s_sorted = s[order]

    # 2. Define the loop body
    # Instead of a massive matrix, we compute one "offset layer" at a time.
    def body_fn(i, val):
        Q_accum = val
        
        # Map loop index i (0 to 2*window) to offset (-window to +window)
        offset = i - window_size
        
        # Fetch Neighbor Data using Roll
        # jnp.roll is O(N) and very fast on GPU
        x_neighbor = jnp.roll(x_sorted, shift=offset, axis=0)
        v_neighbor = jnp.roll(v_sorted, shift=offset, axis=0)
        s_neighbor = jnp.roll(s_sorted, shift=offset, axis=0)

        # Periodic Distance
        dx = x_sorted - x_neighbor
        dx = (dx + box_length / 2) % box_length - box_length / 2
        dist = jnp.abs(dx)
        
        # Physics & Masking
        # We multiply by (offset != 0) to avoid self-interaction
        mask = (dist <= eta) & (dist > 0.0) & (offset != 0)
        
        psi = jnp.maximum(0.0, 1.0 - dist / eta) / eta
        psi = psi * mask

        # Since we are 1D in the loop, we need to expand dims for broadcasting
        # psi is (N,), we need (N, 1) to multiply against (N, d)
        psi = psi[:, None]

        dv = v_sorted - v_neighbor
        ds = s_sorted - s_neighbor
        
        interaction = A_apply_new(dv, ds, gamma)
        
        # Accumulate result
        return Q_accum + interaction * psi

    # 3. Run the Loop
    # We iterate 2*window_size + 1 times.
    # Unlike your original code, the work inside here is perfectly balanced (size N).
    Q_init = jnp.zeros_like(v)
    Q_sorted = lax.fori_loop(0, 2 * window_size + 1, body_fn, Q_init)

    # 4. Unsort
    rev = jnp.empty_like(order).at[order].set(jnp.arange(N))
    return w * Q_sorted[rev]

# on rtx6k, with fp64, n=1e6, M=100, dv=2, takes ~16s and 448Mb memory
def collision_5(x, v, s, eta, gamma, box_length, w):
    """
    Driver function that calculates window size and dispatches to JIT kernel.
    """
    n = v.shape[0]
    
    # 1. Pre-compute the window size (Python int)
    # This is fast and happens outside JAX.
    win_size = compute_window_params(n, eta, box_length)
    
    # 2. Call the JIT-compiled kernel
    return collision_rolling(x, v, s, eta, gamma, box_length, w, window_size=win_size)

@partial(jax.jit, static_argnames=['num_batches', 'spatial_sort'])
def collision_6(x, v, s, eta, gamma, box_length, w, key=jr.PRNGKey(1), num_batches=10, spatial_sort=False):
    """
    Random Batch Method (RBM) Implementation.
    
    Args:
        batch_size: Size of the random batches (P in the text).
        spatial_sort: If True, sorts particles before batching. 
                      Recommended for local kernels (psi_eta) to reduce variance.
    """
    if x.ndim == 2: x = x[:, 0]
    N, d = v.shape
    
    # --- 1. Shuffle / Sort ---
    # The text says "assign them randomly to R batches".
    if spatial_sort:
        # OPTION A: Local RBM (Sorts spatially first)
        # High accuracy for local kernels, biases towards nearest neighbors
        perm = jnp.argsort(x)
    else:
        # OPTION B: Global RBM (True Random)
        # Matches the image text exactly. Unbiased but high variance for local kernels.
        perm = jax.random.permutation(key, N)

    x_shuffled = x[perm]
    v_shuffled = v[perm]
    s_shuffled = s[perm]

    # --- 2. Batching ---
    # We reshape to (Num_Batches, Batch_Size, d)
    # Note: We truncate N to be divisible by batch_size for speed.
    # In a production sim, you'd handle the remainder, but for N=10^6 it's negligible.
    batch_size = N // num_batches
    limit = num_batches * batch_size
    
    x_batches = x_shuffled[:limit].reshape(num_batches, batch_size, 1)
    v_batches = v_shuffled[:limit].reshape(num_batches, batch_size, d)
    s_batches = s_shuffled[:limit].reshape(num_batches, batch_size, d)

    # --- 3. Dense Interaction within Batches ---
    def compute_batch(xb, vb, sb):
        # xb shape: (Batch_Size, 1)
        # Broadcast to (Batch_Size, Batch_Size)
        
        # Distances
        dx = xb - xb.T
        dx = (dx + box_length / 2) % box_length - box_length / 2
        dist = jnp.abs(dx)
        
        # Kernel & Mask
        # Note: We assume interactions within the batch.
        mask = (dist <= eta) & (dist > 0.0)
        
        psi = jnp.maximum(0.0, 1.0 - dist / eta) / eta
        psi = psi * mask
        
        # Physics
        dv = vb[:, None, :] - vb[None, :, :]
        ds = sb[:, None, :] - sb[None, :, :]
        
        interaction = A_apply_new(dv, ds, gamma) # (B, B, d)
        
        # Sum over neighbors (axis 1)
        return jnp.sum(interaction * psi[:, :, None], axis=1)

    # Vectorize over the 'num_batches' dimension
    Q_batches = jax.vmap(compute_batch)(x_batches, v_batches, s_batches)
    
    # Flatten back to (limit, d)
    Q_flat = Q_batches.reshape(limit, d)
    
    # Pad back to N if we truncated
    if limit < N:
        pad = jnp.zeros((N - limit, d))
        Q_flat = jnp.concatenate([Q_flat, pad], axis=0)

    # --- 4. Scaling (The Formula from Image) ---
    # Formula: R * (N - 1) / (N - R)
    # R = num_batches
    scaling_factor = (num_batches * (N - 1)) / (N - num_batches)
    Q_flat = Q_flat * scaling_factor

    # --- 5. Unshuffle ---
    # We need to scatter the forces back to the original particle indices
    rev = jnp.empty_like(perm).at[perm].set(jnp.arange(N))
    return w * Q_flat[rev]
#%%
# prepare data
seed = 42
dx = 1       # Position dimension
dv = 2       # Velocity dimension
k = 0.5
L = 2 * jnp.pi / k   # ~12.566
n = 10**4    # number of particles
M = 100      # number of cells
eta = L / M  # cell size
cells = (jnp.arange(M) + 0.5) * eta

key_x, key_v, key_s = jr.split(jr.PRNGKey(seed), 3)
v = jr.multivariate_normal(key_v, jnp.zeros(dv), jnp.eye(dv), shape=(n,))
v = v * 0.1
x = jr.uniform(key_x, shape=(n,1)) * L
s = jr.multivariate_normal(key_s, jnp.zeros(dv), jnp.eye(dv), shape=(n,))
C = 1.0
gamma = -dv
box_length = L
num_cells = M

#%%
# check accuracy
result_collision = collision(x, v, s, eta, C, gamma, box_length, num_cells)
result_collision_2 = collision_2(x, v, s, eta, C, gamma, box_length, num_cells)
result_collision_3 = collision_3(x, v, s, eta, gamma, num_cells, box_length, L / n)
result_collision_4 = collision_4(x, v, s, eta, gamma, box_length, L / n)
result_collision_5 = collision_5(x, v, s, eta, gamma, box_length, L / n)
result_collision_6 = collision_6(x, v, s, eta, gamma, box_length, L / n, num_batches=1)
result_collision_7 = collision_6(x, v, s, eta, gamma, box_length, L / n, num_batches=5)
abs_diff_2 = jnp.abs(result_collision - result_collision_2)
abs_diff_3 = jnp.abs(result_collision - result_collision_3)
abs_diff_4 = jnp.abs(result_collision - result_collision_4)
abs_diff_5 = jnp.abs(result_collision - result_collision_5)
abs_diff_6 = jnp.abs(result_collision - result_collision_6)
avg_diff_7 = jnp.mean(jnp.abs(result_collision - result_collision_7)**2)**0.5
print(f"Max abs diff collision vs collision_2: {jnp.max(abs_diff_2):.6e}")
print(f"Max abs diff collision vs collision_3: {jnp.max(abs_diff_3):.6e}")
print(f"Max abs diff collision vs collision_4: {jnp.max(abs_diff_4):.6e}")
print(f"Max abs diff collision vs collision_5: {jnp.max(abs_diff_5):.6e}")
print(f"Max abs diff collision vs collision_6: {jnp.max(abs_diff_6):.6e}")
print(f"Mean abs diff collision vs collision_7: {avg_diff_7:.6e}")
#%%
print(f"Number of particles: {n:.0e}, Number of cells: {num_cells}, eta: {eta:.4f}, box_length: {box_length:.4f}")
# benchmark runtime and memory
# bench(collision, x, v, s, eta, C, gamma, box_length, num_cells, name="collision")
# cost(collision, x, v, s, eta, C, gamma, box_length, num_cells, name="collision")
# bench(collision_2, x, v, s, eta, C, gamma, box_length, num_cells, name="collision_2")
# cost(collision_2, x, v, s, eta, C, gamma, box_length, num_cells, name="collision_2")
bench(collision_3, x, v, s, eta, gamma, num_cells, box_length, L / n, name="collision_3")
cost(collision_3, x, v, s, eta, gamma, num_cells, box_length, L / n, name="collision_3")

# bench(collision_4, x, v, s, eta, gamma, box_length, L / n, name="collision_4")
# window_size = compute_window_params(v.shape[0], eta, box_length, bucket_size=100, safety_factor=1.2)
# cost(collision_kernel, x, v, s, eta, gamma, box_length, L / n, window_size=window_size, name="collision_4")

bench(collision_5, x, v, s, eta, gamma, box_length, L / n, name="collision_5")
window_size = compute_window_params(v.shape[0], eta, box_length, bucket_size=100, safety_factor=1.2)
cost(collision_rolling, x, v, s, eta, gamma, box_length, L / n, window_size=window_size, name="collision_5")

bench(collision_6, x, v, s, eta, gamma, box_length, L / n, name="collision_6")
cost(collision_6, x, v, s, eta, gamma, box_length, L / n, num_batches=1000, name="collision_6")