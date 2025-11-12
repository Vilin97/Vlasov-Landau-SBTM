#%%
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import time
from functools import partial
import os

# jax.config.update("jax_enable_x64", True)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
# with fp32, n=4e5, M=100, dv=2, takes ~1.2s and 250Mb memory
# with fp32, n=1e6, M=100, dv=2, takes ~6.4s and 640Mb memory
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
#%%
# prepare data
seed = 42
dx = 1       # Position dimension
dv = 2       # Velocity dimension
k = 0.5
L = 2 * jnp.pi / k   # ~12.566
n = 4*10**5    # number of particles
M = 100      # number of cells
eta = L / M  # cell size
cells = (jnp.arange(M) + 0.5) * eta

key_v, key_x = jr.split(jr.PRNGKey(seed), 2)
v = jr.multivariate_normal(key_v, jnp.zeros(dv), jnp.eye(dv), shape=(n,))
v = v - jnp.mean(v, axis=0)
x = jr.uniform(key_x, shape=(n,1)) * L
s = jr.multivariate_normal(key_v, jnp.zeros(dv), jnp.eye(dv), shape=(n,))

C = 1.0
gamma = -3.0
box_length = L
num_cells = M

#%%
# check accuracy
result_collision = collision(x, v, s, eta, C, gamma, box_length, num_cells)
result_collision_2 = collision_2(x, v, s, eta, C, gamma, box_length, num_cells)
result_collision_3 = collision_3(x, v, s, eta, gamma, num_cells, box_length, L / n)
abs_diff_2 = jnp.abs(result_collision - result_collision_2)
abs_diff_3 = jnp.abs(result_collision - result_collision_3)
print(f"Max abs diff collision vs collision_2: {jnp.max(abs_diff_2):.6e}")
print(f"Max abs diff collision vs collision_3: {jnp.max(abs_diff_3):.6e}")

#%%
# benchmark runtime and memory
bench(collision, x, v, s, eta, C, gamma, box_length, num_cells, name="collision")
cost(collision, x, v, s, eta, C, gamma, box_length, num_cells, name="collision")
bench(collision_2, x, v, s, eta, C, gamma, box_length, num_cells, name="collision_2")
cost(collision_2, x, v, s, eta, C, gamma, box_length, num_cells, name="collision_2")
bench(collision_3, x, v, s, eta, gamma, num_cells, box_length, L / n, name="collision_3")
cost(collision_3, x, v, s, eta, gamma, num_cells, box_length, L / n, name="collision_3")
# %%