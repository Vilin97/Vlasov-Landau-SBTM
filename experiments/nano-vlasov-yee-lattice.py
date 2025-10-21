#%%
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import numpy as np
from scipy.signal import argrelextrema

jax.config.update("jax_enable_x64", True)

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

# ============================
# Yee-grid deposition/interp
# ============================

@jax.jit
def deposit_charge_centers(x, centers, eta, L, q_over_N=1.0):
    """
    ρ_j on centers x_j = j*eta (CIC deposition).
    centers: array of shape (M,) giving center locations (only shape is used)
    """
    M    = centers.shape[0]
    xi   = x / eta                            # fractional index w.r.t. centers
    i0   = jnp.floor(xi).astype(jnp.int32) % M
    frac = xi - jnp.floor(xi)
    i1   = (i0 + 1) % M
    w0, w1 = 1.0 - frac, frac

    counts = jnp.zeros_like(centers) \
               .at[i0].add(w0) \
               .at[i1].add(w1)

    # ρ = (q/N) * L * counts / eta
    return (q_over_N * L / eta) * (counts / x.size)


@jax.jit
def deposit_current_faces(x, vx, E_faces, eta, L, q_over_N=1.0):
    """
    J_{j+1/2} on faces x_{j+1/2} = (j+1/2)*eta (CIC to faces).
    E_faces: array of shape (M,) at faces (only shape is used)
    """
    M    = E_faces.shape[0]
    xi   = x / eta
    iF   = jnp.floor(xi).astype(jnp.int32) % M
    frac = xi - jnp.floor(xi)
    jF   = (iF + 1) % M
    wL, wR = 1.0 - frac, frac

    accum = jnp.zeros_like(E_faces) \
              .at[iF].add(wL * vx) \
              .at[jF].add(wR * vx)

    # J = (q/N) * L * <ψ> * v  with <ψ> ~ 1/eta scaling
    return (q_over_N * L / eta) * (accum / x.size)


@jax.jit
def interp_field_from_faces_to_particles(x, E_faces, eta):
    """
    E(X^p) by linear interpolation from neighboring faces (CIC).
    """
    M    = E_faces.size
    xi   = x / eta
    iF   = jnp.floor(xi).astype(jnp.int32) % M
    frac = xi - jnp.floor(xi)
    jF   = (iF + 1) % M
    return (1.0 - frac) * E_faces[iF] + frac * E_faces[jF]


@jax.jit
def gauss_faces_from_rho_centers(rho_centers, rho_ion, eta):
    """
    Discrete Gauss on Yee grid:
        (E_{j+1/2} - E_{j-1/2})/eta = rho_j - rho_ion
    Build cumulative sum (choose E_{1/2}=0) and then enforce zero-mean on faces.
    """
    src = rho_centers - rho_ion
    E_faces = jnp.cumsum(src) * eta               # faces from centers
    return E_faces - jnp.mean(E_faces)            # fix additive constant

# ----------------------------
# Field & particle updates
# ----------------------------

@jax.jit
def update_electric_field_faces(E_faces, x, v, eta, dt, L, qe_over_N=1.0):
    """
    Ampère on faces (no magnetic term in 1D VA):
        E_{j+1/2}^{n+1} = E_{j+1/2}^n - dt * J_{j+1/2}^n
    """
    J_faces = deposit_current_faces(x, v[:, 0], E_faces, eta, L, q_over_N=qe_over_N)
    E_new = E_faces - dt * J_faces
    # zero-mean preservation (cheap projection)
    return E_new - jnp.mean(E_new)


@jax.jit
def evaluate_field_at_particles(x, E_faces, eta):
    return interp_field_from_faces_to_particles(x, E_faces, eta)


@jax.jit
def step(x, v, E_faces, eta, dt, L, qe_over_N=1.0):
    """
    One explicit Euler step:
      1) interpolate E at particles (faces -> particles)
      2) update v and x
      3) update E on faces via Ampère
    """
    E_p = evaluate_field_at_particles(x, E_faces, eta)
    v_new = v.at[:, 0].add(dt * E_p)
    x_new = jnp.mod(x + dt * v_new[:, 0], L)
    E_new = update_electric_field_faces(E_faces, x, v, eta, dt, L, qe_over_N=qe_over_N)
    return x_new, v_new, E_new

#%%
"Problem setup (Landau damping IC)"

# Physical/IC params
seed  = 42
alpha = 0.1         # perturbation strength
k     = 0.5         # wavenumber
dx    = 1           # position dimension (unused, doc)
dv    = 1           # velocity dimension
qe    = 1.0         # particle charge (absorbed in qe_over_N)
rho_ion = 1.0       # background to ensure neutrality for our normalization

# Particles & grid
num_particles = 10_000_000  # adjust as memory allows
L = 2 * jnp.pi / k
M = 1000
eta = L / M
centers = jnp.arange(M) * eta            # x_j (centers): ρ
faces   = (jnp.arange(M) + 0.5) * eta    # x_{j+1/2} (faces): E

# Velocity ~ N(0, I)
key = jr.PRNGKey(seed)
key_v, key_x = jr.split(key, 2)
v = jr.normal(key_v, shape=(num_particles, dv))
# zero mean velocity:
v = v - jnp.mean(v, axis=0, keepdims=True)

# Spatial density for Landau damping IC: (1 + α cos(kx)) / L (normalized)
def spatial_density(x):
    # normalized to integrate to 1 over [0, L]
    return (1.0 + alpha * jnp.cos(k * x)) / L

# Max value on centers is safe upper bound
max_val = jnp.max(spatial_density(centers))
x = rejection_sample(key_x, spatial_density, (0.0, float(L)), float(max_val), num_samples=num_particles)

# Initial ρ on centers and E on faces (Gauss on Yee)
rho = deposit_charge_centers(x, centers, eta, L, q_over_N=qe)     # centers
E   = gauss_faces_from_rho_centers(rho, rho_ion, eta)             # faces

# Time loop
final_time = 30.0
dt = 0.01
num_steps = int(final_time / dt)
t = 0.0

# Diagnostics
E_L2 = [jnp.sqrt(jnp.sum(E**2) * eta)]

for _ in tqdm(range(num_steps)):
    x, v, E = step(x, v, E, eta, dt, L, qe_over_N=qe)
    # (Optional) enforce zero-mean (already done in update; safe no-op)
    E = E - jnp.mean(E)
    t += dt
    E_L2.append(jnp.sqrt(jnp.sum(E**2) * eta))

#%%
"Plot L2 norm of E over time"
plt.figure(figsize=(6,4))
plt.plot(jnp.linspace(0, final_time, num_steps+1), E_L2, marker='o', markersize=1, label='Simulation')

# Predicted curve
t_grid = jnp.linspace(0, final_time, num_steps+1)
prefactor = - 1/(k**3) * jnp.sqrt(jnp.pi/8) * jnp.exp(-1/(2*k**2) - 1.5)
predicted = jnp.exp(t_grid * prefactor)
predicted *= E_L2[0]/predicted[0]
gamma = prefactor
plt.plot(t_grid, predicted, 'r--', label=fr'$e^{{\gamma t}}, \gamma = {gamma:.3f}$')

# Fit in log space
t_grid = np.asarray(t_grid)
E_L2 = np.asarray(E_L2)
mask = (t_grid > 0.2) & (t_grid < 15)
t_mask = t_grid[mask]
n_mask = E_L2[mask]

maxima_indices = argrelextrema(n_mask, np.greater, order=30)[0]
mt = t_mask[maxima_indices]
mv = n_mask[maxima_indices]
plt.scatter(mt, mv, color='g', marker='o', zorder=5)
coeffs = np.polyfit(mt, np.log(mv), 1)
fit = np.exp(coeffs[1] + coeffs[0] * t_mask)
plt.plot(t_mask, fit, 'g--', label=fr'$e^{{\beta t}}, \beta={coeffs[0]:.3f}$')

plt.xlabel('Time')
plt.ylabel(r'$||E||_{L^2}$')
plt.title(f"n={num_particles:.0e}, Δt={dt}, dv={dv}, α={alpha}, C=0, N={M}")
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# %%
