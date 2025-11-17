#%%
"Optimize the score_kde functions"
import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
import jax.lax as lax
import time
jax.config.update("jax_enable_x64", True) # float64 is ~12x slower than float32

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
def _silverman_bandwidth(v, eps=1e-12):
    n, dv = v.shape
    sigma = jnp.std(v, axis=0, ddof=1) + eps
    return sigma * n ** (-1.0 / (dv + 4.0))  # (dv,)

def _periodic_dx(x, L):
    dx = x[:, None] - x[None, :]
    return (dx + 0.5 * L) % L - 0.5 * L

# Baseline: naive implementation, O(n^2)
@jax.jit
def score_kde_naive(x, v, cells, eta, eps=1e-12, hv=None):
    L = eta * cells.size
    if hv is None:
        hv = _silverman_bandwidth(v, eps)
    dx = _periodic_dx(x, L)                           # (n,n)
    psi = jnp.maximum(0.0, 1.0 - jnp.abs(dx) / eta)   # hat kernel
    dv = v[:, None, :] - v[None, :, :]                # (n,n,dv)
    K = jnp.exp(-0.5 * jnp.sum((dv / hv) ** 2, axis=-1))  # (n,n)
    w  = psi * K + eps
    Z  = jnp.sum(w, axis=1, keepdims=True)
    mu = (w @ v) / Z                                  # (n,dv)
    return (mu - v) / (hv ** 2)

# Baseline: vmap over particles, O(n^2)
@jax.jit
def kde_score_vmap(x, v, eta_x, eta_v, L=None):
    """∇_v log f for 1-D (x,v) samples via KDE with hat kernels."""
    def psi_hat(u, eta, L=None):
        if L is not None:                       # periodic distance
            u = (u + 0.5 * L) % L - 0.5 * L
        return jnp.maximum(0., 1. - jnp.abs(u) / eta) / eta

    eta_x, eta_v = map(jnp.asarray, (eta_x, eta_v))  # make them tracers
    eps = 1e-12

    def score_single(xi, vi):
        def log_f(vi_):
            w_x = psi_hat(xi - x, eta_x, L)          # (N,)
            w_v = psi_hat(vi_ - v, eta_v)            # (N,)
            return jnp.log(jnp.mean(w_x * w_v) + eps)
        return jax.grad(log_f)(vi)                   # scalar grad

    return jax.vmap(score_single)(x, v)              # (N,)

# Optimized for memory: stream over particles
@partial(jax.jit, static_argnames=['jchunk'])
def score_kde_stream(x, v, cells, eta, eps=1e-12, hv=None, jchunk=2048):
    L = eta * cells.size
    if hv is None: hv = _silverman_bandwidth(v, eps)
    inv_hv2 = 1.0 / (hv**2)

    n, dv = v.shape
    n_it = (n + jchunk - 1) // jchunk
    n_pad = n_it * jchunk
    pad = n_pad - n

    x_pad = jnp.pad(x, (0, pad))
    v_pad = jnp.pad(v, ((0, pad), (0, 0)))

    Z = jnp.zeros((n, 1), v.dtype)
    M = jnp.zeros((n, dv), v.dtype)

    ar = jnp.arange(jchunk)  # for masking

    def body(t, carry):
        Zc, Mc = carry
        j0 = t * jchunk

        xj = lax.dynamic_slice(x_pad, (j0,), (jchunk,))         # (m,)
        vj = lax.dynamic_slice(v_pad, (j0, 0), (jchunk, dv))    # (m,dv)

        # periodic dx against chunk
        dx = x[:, None] - xj[None, :]
        dx = (dx + 0.5 * L) % L - 0.5 * L
        psi = jnp.clip(1.0 - jnp.abs(dx) / eta, 0.0, 1.0)       # (n,m)

        dv_ = v[:, None, :] - vj[None, :, :]                    # (n,m,dv)
        Kj = jnp.exp(-0.5 * jnp.sum((dv_ * jnp.sqrt(inv_hv2))**2, axis=-1))  # (n,m)

        m_eff = jnp.minimum(jchunk, n - j0)
        mask = (ar < m_eff)[None, :]                            # (1,m)
        w = (psi * Kj + eps) * mask

        Zc += jnp.sum(w, axis=1, keepdims=True)
        Mc += w @ vj
        return Zc, Mc

    Z, M = lax.fori_loop(0, n_it, body, (Z, M))
    mu = M / Z
    return (mu - v) * inv_hv2

# THIS IS THE BEST VERSION
# on l40s with fp32, n=4e5, M=100, dv=2, takes ~6s and 70Mb memory
# on l40s with fp32, n=1e6, M=100, dv=2, takes ~38s and 120Mb memory
@partial(jax.jit, static_argnames=['ichunk', 'jchunk'])
def score_kde_blocked(x, v, cells, eta, eps=1e-12, hv=None, ichunk=2048, jchunk=2048):
    L = eta * cells.size
    if hv is None: hv = _silverman_bandwidth(v, eps)
    n, dv = v.shape
    inv_hv = 1.0 / hv
    inv_hv2 = inv_hv**2

    ni = (n + ichunk - 1) // ichunk
    nj = (n + jchunk - 1) // jchunk
    n_pad = ni * ichunk
    pad = n_pad - n

    x_pad = jnp.pad(x, (0, pad))
    v_pad = jnp.pad(v, ((0, pad), (0, 0)))
    u_pad = v_pad * inv_hv

    Zp = jnp.zeros((n_pad, 1), v.dtype)
    Mp = jnp.zeros((n_pad, dv), v.dtype)

    ar_i = jnp.arange(ichunk)
    ar_j = jnp.arange(jchunk)

    def loop_j(tj, carry2):
        Zi_, Mi_, Ri, Ui, Vi, Ui2 = carry2
        j0 = tj * jchunk
        mj = jnp.minimum(jchunk, n - j0)

        Rj = lax.dynamic_slice(x_pad, (j0,), (jchunk,))
        Uj = lax.dynamic_slice(u_pad, (j0, 0), (jchunk, dv))
        Vj = lax.dynamic_slice(v_pad, (j0, 0), (jchunk, dv))
        Uj2 = jnp.sum(Uj * Uj, axis=1, keepdims=True).T
        mask_j = (ar_j < mj).astype(v.dtype).reshape(1, jchunk)

        dx = Ri[:, None] - Rj[None, :]
        dx = (dx + 0.5 * L) % L - 0.5 * L
        psi = jnp.clip(1.0 - jnp.abs(dx) / eta, 0.0, 1.0)

        G = Ui @ Uj.T
        Kj = jnp.exp(G - 0.5 * Ui2 - 0.5 * Uj2)

        w = (psi * Kj + eps) * mask_j
        Zi_ = Zi_ + jnp.sum(w, axis=1, keepdims=True)
        Mi_ = Mi_ + w @ Vj
        return Zi_, Mi_, Ri, Ui, Vi, Ui2

    def loop_i(ti, carry):
        Zc, Mc = carry
        i0 = ti * ichunk
        mi = jnp.minimum(ichunk, n - i0)

        Ri = lax.dynamic_slice(x_pad, (i0,), (ichunk,))
        Ui = lax.dynamic_slice(u_pad, (i0, 0), (ichunk, dv))
        Vi = lax.dynamic_slice(v_pad, (i0, 0), (ichunk, dv))
        Ui2 = jnp.sum(Ui * Ui, axis=1, keepdims=True)

        Zi = jnp.zeros((ichunk, 1), v.dtype)
        Mi = jnp.zeros((ichunk, dv), v.dtype)

        Zi, Mi, *_ = lax.fori_loop(0, nj, loop_j, (Zi, Mi, Ri, Ui, Vi, Ui2))

        mask_i = (ar_i < mi).astype(v.dtype).reshape(ichunk, 1)
        Zi = Zi * mask_i
        Mi = Mi * mask_i

        Zc = lax.dynamic_update_slice(Zc, Zi, (i0, 0))
        Mc = lax.dynamic_update_slice(Mc, Mi, (i0, 0))
        return Zc, Mc

    Zp, Mp = lax.fori_loop(0, ni, loop_i, (Zp, Mp))
    Z = Zp[:n]
    M = Mp[:n]
    mu = M / Z
    return (mu - v) * inv_hv2

@partial(jax.jit, static_argnames=['max_ppc'])
def _score_kde_local_impl(x, v, cells, eta, eps=1e-12, hv=None, max_ppc=4096):
    if x.ndim == 2:
        x = x[:, 0]
    if hv is None:
        hv = _silverman_bandwidth(v, eps)

    n, dv = v.shape
    M = cells.size
    L = eta * M
    inv_hv = 1.0 / hv
    inv_hv2 = inv_hv ** 2

    idx = jnp.floor(x / eta).astype(jnp.int32) % M
    order = jnp.argsort(idx)
    x_s = x[order]
    v_s = v[order]
    idx_s = idx[order]

    counts = jnp.bincount(idx_s, length=M)
    cell_ofs = jnp.cumsum(
        jnp.concatenate([jnp.array([0], dtype=jnp.int32), counts[:-1]])
    )

    Xc = jnp.zeros((M, max_ppc), x.dtype)
    Vc = jnp.zeros((M, max_ppc, dv), v.dtype)
    maskc = jnp.zeros((M, max_ppc), x.dtype)
    idx_map = -jnp.ones((M, max_ppc), jnp.int32)
    ar_ppc = jnp.arange(max_ppc, dtype=jnp.int32)

    def fill_cell(c, carry):
        Xc, Vc, maskc, idx_map = carry
        cnt = counts[c]
        base = cell_ofs[c]
        valid = ar_ppc < cnt
        gidx = base + ar_ppc
        gidx = jnp.where(valid, gidx, 0)
        Xc = Xc.at[c].set(jnp.where(valid, x_s[gidx], 0.0))
        Vc = Vc.at[c].set(jnp.where(valid[:, None], v_s[gidx], 0.0))
        maskc = maskc.at[c].set(valid.astype(x.dtype))
        idx_map = idx_map.at[c].set(jnp.where(valid, gidx, -1))
        return Xc, Vc, maskc, idx_map

    Xc, Vc, maskc, idx_map = lax.fori_loop(
        0, M, fill_cell, (Xc, Vc, maskc, idx_map)
    )

    Uc = Vc * inv_hv
    U2c = jnp.sum(Uc * Uc, axis=-1, keepdims=True)

    Zc = jnp.zeros((M, max_ppc, 1), v.dtype)
    Mc = jnp.zeros((M, max_ppc, dv), v.dtype)

    def body_cell(c, carry):
        Zc, Mc = carry
        Xi = Xc[c]                    # (max_ppc,)
        Vi = Vc[c]                    # (max_ppc,dv)
        Ui = Uc[c]
        Ui2 = U2c[c]                  # (max_ppc,1)
        mask_i = maskc[c][:, None]    # (max_ppc,1)

        c0 = (c - 1) % M
        c1 = c
        c2 = (c + 1) % M
        Xj = jnp.concatenate([Xc[c0], Xc[c1], Xc[c2]], axis=0)        # (3*max_ppc,)
        Vj = jnp.concatenate([Vc[c0], Vc[c1], Vc[c2]], axis=0)        # (3*max_ppc,dv)
        Uj = jnp.concatenate([Uc[c0], Uc[c1], Uc[c2]], axis=0)
        Uj2 = jnp.concatenate([U2c[c0], U2c[c1], U2c[c2]], axis=0)    # (3*max_ppc,1)
        mask_j = jnp.concatenate(
            [maskc[c0], maskc[c1], maskc[c2]], axis=0
        )[:, None]                                                     # (3*max_ppc,1)

        dx = Xi[:, None] - Xj[None, :]
        dx = (dx + 0.5 * L) % L - 0.5 * L
        psi = jnp.maximum(0.0, 1.0 - jnp.abs(dx) / eta)               # hat in x

        G = Ui @ Uj.T
        K = jnp.exp(G - 0.5 * Ui2 - 0.5 * Uj2.T)

        mask = mask_i * mask_j.T
        w = (psi * K + eps) * mask

        Z_local = jnp.sum(w, axis=1, keepdims=True) * mask_i
        M_local = (w @ Vj) * mask_i

        Zc = Zc.at[c].set(Z_local)
        Mc = Mc.at[c].set(M_local)
        return Zc, Mc

    Zc, Mc = lax.fori_loop(0, M, body_cell, (Zc, Mc))

    idx_flat = idx_map.reshape(-1)
    Z_flat = Zc.reshape(-1, 1)
    M_flat = Mc.reshape(-1, dv)
    valid = idx_flat >= 0
    idx_valid = jnp.where(valid, idx_flat, 0)
    Z_contrib = Z_flat * valid[:, None]
    M_contrib = M_flat * valid[:, None]

    Zs = jnp.zeros((n, 1), v.dtype)
    Ms = jnp.zeros((n, dv), v.dtype)
    Zs = Zs.at[idx_valid].add(Z_contrib)
    Ms = Ms.at[idx_valid].add(M_contrib)

    inv_order = jnp.empty_like(order)
    inv_order = inv_order.at[order].set(jnp.arange(n))

    Z = Zs[inv_order]
    M = Ms[inv_order]
    
    Z_safe = jnp.where(Z > 0, Z, eps)
    mu = M / Z_safe
    jax.debug.print("max_ppc={max_ppc}, max_count={mc}", max_ppc=max_ppc, mc=jnp.max(counts))
    return (mu - v) * inv_hv2

# this is ~11 times faster than score_kde_blocked with n=1e5 and M=50
def score_kde_local(x, v, cells, eta, eps=1e-12, hv=None):
    if hv is None:
        hv = _silverman_bandwidth(v, eps)

    if x.ndim == 2:
        x1d = x[:, 0]
    else:
        x1d = x
    M = cells.size
    idx = jnp.floor(x1d / eta).astype(jnp.int32) % M
    counts = jnp.bincount(idx, length=M)
    max_count = int(jax.device_get(jnp.max(counts)))
    m = max(1, max_count)
    max_ppc = ((m + 99) // 100) * 100  # next multiple of 100 >= m

    return _score_kde_local_impl(x, v, cells, eta, eps, hv, max_ppc)

def _silverman_bandwidth_1d(x, eps=1e-12):
    n = x.size
    sigma = jnp.std(x, ddof=1) + eps
    return sigma * n ** (-1.0 / 5.0)  # d=1

# ~20% faster than score_kde_blocked
@partial(jax.jit, static_argnames=['ichunk', 'jchunk'])
def score_kde_gaussx(x, v, cells, eta, eps=1e-12, hv=None, hx=None,
              ichunk=2048, jchunk=2048):
    if x.ndim == 2:
        x = x[:, 0]
    if hv is None:
        hv = _silverman_bandwidth(v, eps)
    if hx is None:
        hx = _silverman_bandwidth_1d(x, eps)

    L = eta * cells.size
    n, dv = v.shape
    inv_hv = 1.0 / hv
    inv_hv2 = inv_hv**2
    inv_hx2 = 1.0 / (hx**2)

    ni = (n + ichunk - 1) // ichunk
    nj = (n + jchunk - 1) // jchunk
    n_pad = ni * ichunk
    pad = n_pad - n

    x_pad = jnp.pad(x, (0, pad))
    v_pad = jnp.pad(v, ((0, pad), (0, 0)))
    u_pad = v_pad * inv_hv

    Zp = jnp.zeros((n_pad, 1), v.dtype)
    Mp = jnp.zeros((n_pad, dv), v.dtype)

    ar_i = jnp.arange(ichunk)
    ar_j = jnp.arange(jchunk)

    def loop_j(tj, carry2):
        Zi_, Mi_, Ri, Ui, Vi, Ui2 = carry2
        j0 = tj * jchunk
        mj = jnp.minimum(jchunk, n - j0)

        Rj = lax.dynamic_slice(x_pad, (j0,), (jchunk,))
        Uj = lax.dynamic_slice(u_pad, (j0, 0), (jchunk, dv))
        Vj = lax.dynamic_slice(v_pad, (j0, 0), (jchunk, dv))
        Uj2 = jnp.sum(Uj * Uj, axis=1, keepdims=True).T
        mask_j = (ar_j < mj).astype(v.dtype).reshape(1, jchunk)

        dx = Ri[:, None] - Rj[None, :]
        dx = (dx + 0.5 * L) % L - 0.5 * L
        Kx = jnp.exp(-0.5 * (dx * dx) * inv_hx2)  # Gaussian in x (periodic)

        G  = Ui @ Uj.T
        Kv = jnp.exp(G - 0.5 * Ui2 - 0.5 * Uj2)

        w = (Kx * Kv + eps) * mask_j
        Zi_ = Zi_ + jnp.sum(w, axis=1, keepdims=True)
        Mi_ = Mi_ + w @ Vj
        return Zi_, Mi_, Ri, Ui, Vi, Ui2

    def loop_i(ti, carry):
        Zc, Mc = carry
        i0 = ti * ichunk
        mi = jnp.minimum(ichunk, n - i0)

        Ri = lax.dynamic_slice(x_pad, (i0,), (ichunk,))
        Ui = lax.dynamic_slice(u_pad, (i0, 0), (ichunk, dv))
        Vi = lax.dynamic_slice(v_pad, (i0, 0), (ichunk, dv))
        Ui2 = jnp.sum(Ui * Ui, axis=1, keepdims=True)

        Zi = jnp.zeros((ichunk, 1), v.dtype)
        Mi = jnp.zeros((ichunk, dv), v.dtype)

        Zi, Mi, *_ = lax.fori_loop(0, nj, loop_j, (Zi, Mi, Ri, Ui, Vi, Ui2))

        mask_i = (ar_i < mi).astype(v.dtype).reshape(ichunk, 1)
        Zi = Zi * mask_i
        Mi = Mi * mask_i

        Zc = lax.dynamic_update_slice(Zc, Zi, (i0, 0))
        Mc = lax.dynamic_update_slice(Mc, Mi, (i0, 0))
        return Zc, Mc

    Zp, Mp = lax.fori_loop(0, ni, loop_i, (Zp, Mp))
    Z = Zp[:n]
    M = Mp[:n]
    mu = M / Z
    return (mu - v) * inv_hv2

#%%
# prepare data
seed = 42
dx = 1       # Position dimension
dv = 2       # Velocity dimension
k = 0.5
L = 2 * jnp.pi / k   # ~12.566
n = 10**6    # number of particles
M = 100      # number of cells
eta = L / M  # cell size
cells = (jnp.arange(M) + 0.5) * eta

key_v, key_x = jr.split(jr.PRNGKey(seed), 2)
v = jr.multivariate_normal(key_v, jnp.zeros(dv), jnp.eye(dv), shape=(n,))
v = v - jnp.mean(v, axis=0)
x = jr.uniform(key_x, shape=(n,)) * L

#%%
# benchmark runtime and memory

# s_naive = score_kde_naive(x, v, cells, eta)
# s_stream = score_kde_stream(x, v, cells, eta)
# s_blocked = score_kde_blocked(x, v, cells, eta)
# s_local = score_kde_local(x, v, cells, eta)
# print("Max abs diff stream vs naive:", jnp.max(jnp.abs(s_stream - s_naive)))
# print("Max abs diff blocked vs naive:", jnp.max(jnp.abs(s_blocked - s_naive)))
# print(f"Max abs diff blocked vs local: {jnp.max(jnp.abs(s_blocked - s_local)):.6e}")

# bench(score_kde_naive, x, v, cells, eta, name="naive")
# cost(score_kde_naive, x, v, cells, eta, name="naive")

# bench(kde_score_vmap, x, v[:,0], eta, eta, L, name="kde_score_vmap")
# cost(kde_score_vmap, x, v[:,0], eta, eta, L, name="kde_score_vmap")

# bench(score_kde_stream, x, v, cells, eta, name=f"stream")
# cost(score_kde_stream, x, v, cells, eta, name=f"stream")

bench(score_kde_blocked, x, v, cells, eta, name=f"blocked")
cost(score_kde_blocked, x, v, cells, eta, name=f"blocked")

bench(score_kde_local, x, v, cells, eta, name=f"local")
cost(_score_kde_local_impl, x, v, cells, eta, name=f"local")

bench(score_kde_gaussx, x, v, cells, eta, name=f"gaussx")
cost(score_kde_gaussx, x, v, cells, eta, name=f"gaussx")

# %%
s_blocked = score_kde_blocked(x, v, cells, eta)
s_local = score_kde_local(x, v, cells, eta)
abs_diff_local = jnp.abs(s_blocked - s_local)
print(f"Max abs diff blocked vs local: {jnp.max(abs_diff_local):.6e}")
# %%
bench(score_kde_local, x, v, cells, eta, name=f"local")
cost(_score_kde_local_impl, x, v, cells, eta, name=f"local")