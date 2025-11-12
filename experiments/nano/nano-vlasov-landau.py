#%%
"Self-contained Vlasov-Landau solver"

import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
import matplotlib.pyplot as plt
import jax.lax as lax
from functools import partial

# jax.config.update("jax_enable_x64", True)

def visualize_initial(x, v, cells, E, rho, eta, L, v_target=lambda v: jax.scipy.stats.norm.pdf(v, 0, 1)):
    """Visualize initial data."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Histogram of x and desired density
    axs[0].hist(x, bins=50, density=True, alpha=0.4, label='Sampled $x$')
    x_grid = jnp.linspace(0, L, 200)
    axs[0].plot(x_grid, spatial_density(x_grid), 'r-', label='Target density')
    axs[0].plot(cells, rho / L, 'g-', label='$\\rho/L$')
    axs[0].set_title('Position $x$')
    axs[0].set_xlabel('$x$')
    axs[0].legend()

    # 2. Histogram of v and standard normal
    axs[1].hist(v, bins=50, density=True, alpha=0.4, label='Sampled $v$')
    v_grid = jnp.linspace(v.min()-1, v.max()+1, 200)
    axs[1].plot(v_grid, v_target(v_grid), 'r-', label='Target $N(0,1)$')
    axs[1].set_title('Velocity $v$')
    axs[1].set_xlabel('$v$')
    axs[1].legend()

    # 3. E, dE/dx, and rho
    axs[2].plot(cells, E, label='$E$')
    dE_dx = jnp.gradient(E, eta)
    axs[2].plot(cells, dE_dx, label='$dE/dx$')
    axs[2].plot(cells, rho - 1, label=r'$\rho - \rho_i$')
    axs[2].set_title('Field $E$, $dE/dx$, and $\\rho$')
    axs[2].set_xlabel('x')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def rejection_sample(key, density_fn, domain, max_value, num_samples=1):
    "sample in parallel"

    domain_width = domain[1] - domain[0]
    proposal_fn = lambda x: jnp.where((x >= domain[0]) & (x <= domain[1]), 1.0 / domain_width, 0.0)
    max_ratio = max_value / (1.0 / domain_width) * 1.2 # 20% margin
    key, key_propose, key_accept = jr.split(key, 3)

    # sample twice the needed-in-expectation amount
    num_candidates = int(num_samples * max_ratio * 2)
    candidates = jr.uniform(key_propose, minval=domain[0], maxval=domain[1], shape=(num_candidates,))
    proposal_values = proposal_fn(candidates)
    target_values = density_fn(candidates)
    
    # Accept with probability target/(proposal * max_ratio)
    accepted = jr.uniform(key_accept, num_candidates) * max_ratio * proposal_values <= target_values
    samples = candidates[accepted]
    
    return samples[:num_samples]

#%%
#------------------------------------------------------------------------------
# PIC pieces (1D x, 2D v)
#------------------------------------------------------------------------------
@jax.jit
def evaluate_charge_density(x, cells, eta, w):
    """ ρ_j = w * Σ_p ψ_eta(X_p − cell_j)   with ψ the hat kernel. """
    M = cells.size
    idx_f = x / eta - 0.5
    i0 = jnp.floor(idx_f).astype(jnp.int32) % M
    i1 = (i0 + 1) % M
    f = idx_f - jnp.floor(idx_f)
    w0, w1 = 1 - f, f
    counts = (
        jnp.zeros(M)
          .at[i0].add(w0)
          .at[i1].add(w1)
    )
    return w / eta * counts

@jax.jit
def evaluate_field_at_particles(E, x, cells, eta):
    """ E(x) = Σ_j ψ(x − cell_j) E_j   (linear-hat kernel, periodic) """
    M = cells.size
    idx_f = x / eta - 0.5
    i0 = jnp.floor(idx_f).astype(jnp.int32) % M
    f = idx_f - jnp.floor(idx_f)
    i1 = (i0 + 1) % M
    return (1.0 - f) * E[i0] + f * E[i1]

@jax.jit
def update_electric_field(E, x, v, cells, eta, w, dt):
    """ E_j^{n+1} = E_j^n - dt * w * Σ_i ψ(x_i - cell_j) v_i   (linear-hat kernel, periodic) """
    M = cells.size
    idx_f = x / eta - 0.5
    i0 = jnp.floor(idx_f).astype(jnp.int32) % M
    i1 = (i0 + 1) % M
    f = idx_f - jnp.floor(idx_f)
    w0, w1 = 1 - f, f
    J = (
        jnp.zeros(M)
          .at[i0].add(w0 * v[:, 0])
          .at[i1].add(w1 * v[:, 0])
    )
    dEdt = w / eta * J
    return (E - dt * dEdt).astype(E.dtype)

@jax.jit
def vlasov_step(x, v, E, cells, eta, dt, box_length, w):
    "Forward Euler time stepping"
    E_at_particles = evaluate_field_at_particles(E, x, cells, eta)
    v_new = v.at[:, 0].add(dt * E_at_particles)
    x_new = jnp.mod(x + dt * v[:, 0], box_length)
    E_new = update_electric_field(E, x, v, cells, eta, w, dt)
    return x_new, v_new, E_new

#------------------------------------------------------------------------------
# KDE in phase space and Landau collision
#------------------------------------------------------------------------------
def _silverman_bandwidth(v, eps=1e-12):
    n, dv = v.shape
    sigma = jnp.std(v, axis=0, ddof=1) + eps
    return sigma * n ** (-1.0 / (dv + 4.0))  # (dv,)

@partial(jax.jit, static_argnames=['ichunk', 'jchunk'])
def score_kde(x, v, cells, eta, eps=1e-12, hv=None, ichunk=2048, jchunk=2048):
    if x.ndim == 2:
        x = x[:, 0]
    if hv is None: hv = _silverman_bandwidth(v, eps)
    L = eta * cells.size
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

def sd_score_kde(x, v, cells, eta, eps=1e-12, hv=None, ichunk=2048, jchunk=2048):
    """
    Score-debiased KDE score estimator
    Computes KDE(x,v + hv^2/2 s(x,v))
    """
    if hv is None: hv = _silverman_bandwidth(v, eps)
    s_kde = score_kde(x, v, cells, eta, eps, hv, ichunk, jchunk)
    v_sd = v + (hv ** 2) / 2 * s_kde
    return score_kde(x, v_sd, cells, eta, eps, hv, ichunk, jchunk)

def scaled_score_kde(x, v, cells, eta, eta_scale=4, hv_scale=4, output_scale=1.3, **kwargs):
    "Empirically found scalings for better accuracy"
    hv=_silverman_bandwidth(v) * hv_scale
    s_kde = score_kde(x, v, cells, eta*eta_scale, hv=hv, **kwargs) * output_scale
    return s_kde

#------------------------------------------------------------------------------
# Landau collision operator
#------------------------------------------------------------------------------
@jax.jit
def A_apply(dv, ds, gamma, eps=1e-14):
    v2 = jnp.sum(dv * dv, axis=-1, keepdims=True) + eps
    vg = v2 ** (gamma / 2)
    dvds = jnp.sum(dv * ds, axis=-1, keepdims=True)
    return vg * (v2 * ds - dvds * dv)

@partial(jax.jit, static_argnames=['num_cells'])
def collision(x, v, s, eta, gamma, num_cells, box_length, w):
    """ 
    Q_i = w Σ_{|x_i−x_j|≤η} ψ(x_i−x_j) A(v_i−v_j)(s_i−s_j) with the linear-hat kernel ψ of width eta, periodic on [0,L]. 
    Complexity O(N η/L). 
    """
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
#------------------------------------------------------------------------------
# set up initial data
#------------------------------------------------------------------------------
seed = 42
q = 1        # particle charge
dx = 1       # Position dimension
dv = 2       # Velocity dimension
alpha = 0.1  # amplitude of initial density perturbation
k = 0.5
L = 2 * jnp.pi / k   # ~12.566
n = 10**5    # number of particles
M = 100       # number of cells
dt = 0.02    # time step
eta = L / M  # cell size
cells = (jnp.arange(M) + 0.5) * eta
w = q * L / n # particle weight
C = 0.05     # collision strength
gamma = -dv  # Coulomb interaction

key_v, key_x = jr.split(jr.PRNGKey(seed), 2)
v = jr.multivariate_normal(key_v, jnp.zeros(dv), jnp.eye(dv), shape=(n,))
v = v - jnp.mean(v, axis=0)

def spatial_density(x):
    return (1 + alpha * jnp.cos(k * x)) / (2 * jnp.pi / k)

max_value = jnp.max(spatial_density(cells))
domain = (0.0, L)
x = rejection_sample(key_x, spatial_density, domain, max_value=max_value, num_samples=n)

rho = evaluate_charge_density(x, cells, eta, w)
E = jnp.cumsum(rho - 1) * eta
E = E - jnp.mean(E)
visualize_initial(x, v[:, 0], cells, E, rho, eta, L)

final_time = 15.0
num_steps = int(final_time / dt)
t = 0.0
E_L2 = [jnp.sqrt(jnp.sum(E**2) * eta)]

#%%
s_kde = scaled_score_kde(x, v, cells, eta)
s_true = -v

# Subsample for readability
step = max(1, n // 500)
v_plot = v[::step]
s_kde_plot = s_kde[::step]
s_true_plot = s_true[::step]

plt.figure(figsize=(6, 6))
q2 = plt.quiver(
    v_plot[:, 0],
    v_plot[:, 1],
    s_kde_plot[:, 0],
    s_kde_plot[:, 1],
    color="tab:blue",
    alpha=0.8,
    scale=5,
    angles='xy',
    scale_units='xy',
    label=f"KDE score n={n:.1e} mse={float(jnp.mean((s_kde - s_true)**2)):.3f}"
)
q2 = plt.quiver(
    v_plot[:, 0],
    v_plot[:, 1],
    s_true_plot[:, 0],
    s_true_plot[:, 1],
    color="tab:red",
    alpha=0.5,
    scale=5,
    angles='xy',
    scale_units='xy',
    label="True score (-v)"
)
plt.scatter(v_plot[:, 0], v_plot[:, 1], s=2, c="k", alpha=0.3, label="v samples")
plt.axis("equal")
plt.xlabel("v1")
plt.ylabel("v2")
plt.title("Velocity-space scores: KDE vs True")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

#%%
"main loop"
for _ in tqdm(range(num_steps)):
    # vlasov step
    x, v, E = vlasov_step(x, v, E, cells, eta, dt, L, w)
    
    # collision step
    s = scaled_score_kde(x, v, cells, eta)
    Q = collision(x, v, s, eta, gamma, n, L, w)
    v = v - dt * C * Q

    E = E - jnp.mean(E)
    t += dt
    E_L2.append(jnp.sqrt(jnp.sum(E**2) * eta))
# %%
import numpy as np
import jax.numpy as jnp
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

t_grid = jnp.linspace(0, final_time, num_steps+1)

plt.figure(figsize=(6,4))

# 1) raw sims
plt.plot(t_grid, E_L2,  marker='o', ms=1, label=f'Simulation (C={C})')

# 2) predicted curve for collisionless
prefactor = -1/(k**3) * jnp.sqrt(jnp.pi/8) * jnp.exp(-1/(2*k**2) - 1.5)
pred = jnp.exp(t_grid * prefactor)
pred *= E_L2[0] / pred[0]
plt.plot(t_grid, pred, 'k-.', label=fr'collisionless: $e^{{\beta t}}, \beta={float(prefactor):.3f}$')

# 3) predicted curve for collisional
prefactor = -1/(k**3) * jnp.sqrt(jnp.pi/8) * jnp.exp(-1/(2*k**2) - 1.5)
prefactor_collisional = prefactor - C * jnp.sqrt(2/(9*jnp.pi))
predicted_collisional = jnp.exp(t_grid * prefactor_collisional)
predicted_collisional *= E_L2[0] / predicted_collisional[0]
if C > 0:
    plt.plot(t_grid, predicted_collisional, 'r--', label=fr'collisional: $e^{{\beta t}},\ \beta = {prefactor_collisional:.3f}$')

# 4) fit on maxima
t_np = np.asarray(t_grid)
E_np = np.asarray(E_L2)
mask = (t_np > 0.2) & (t_np < 15)
t_mask = t_np[mask]
E_mask = E_np[mask]
max_idx = argrelextrema(E_mask, np.greater, order=20)[0]
mt, mv = t_mask[max_idx], E_mask[max_idx]
plt.scatter(mt, mv, c='g', zorder=5)
coeffs = np.polyfit(mt, np.log(mv), 1)
fit = np.exp(coeffs[1] + coeffs[0]*t_mask)
plt.plot(t_mask, fit, 'g--', label=fr'fitted: $e^{{\beta t}}, \beta={coeffs[0]:.3f}$')

plt.xlabel('Time')
plt.ylabel(r'$||E||_{L^2}$')
plt.title(f"C={C}, n={n:.0e}, Δt={dt}, dv={dv}, α={alpha}, M={M}")
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%
