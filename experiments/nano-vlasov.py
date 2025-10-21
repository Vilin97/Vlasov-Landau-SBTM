#%%
"Self-contained Vlasov solver"

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import numpy as np
from scipy.signal import argrelextrema

jax.config.update("jax_enable_x64", True)

# Visualize initial data
def visualize_initial(x, v, cells, E, rho, eta, L):
    """Visualize initial data."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Histogram of x and desired density
    axs[0].hist(x, bins=50, density=True, alpha=0.6, label='Sampled $x$')
    x_grid = jnp.linspace(0, L, 200)
    axs[0].plot(x_grid, spatial_density(x_grid), 'r-', label='Target density')
    axs[0].plot(cells, rho, 'g-', label='$\\rho$')
    axs[0].set_title('Position $x$')
    axs[0].set_xlabel('$x$')
    axs[0].legend()

    # 2. Histogram of v and standard normal
    axs[1].hist(v, bins=50, density=True, alpha=0.6, label='Sampled $v$')
    v_grid = jnp.linspace(v.min()-1, v.max()+1, 200)
    axs[1].plot(v_grid, jax.scipy.stats.norm.pdf(v_grid, 0, 1), 'r-', label='Target $N(0,1)$')
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

@jax.jit
def update_electric_field(E, cells, x, v, eta, dt, L):
    """
    E_j^{n+1} = E_j^n - dt * L * Σ_i ψ(x_i - cell_j) v_i   (linear-hat kernel, periodic)
    """
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
    return (E - dt * L * J).astype(E.dtype)

@jax.jit
def evaluate_field_at_particles(x, cells, E, eta, L):
    """
    Σ_j ψ(x_i − cell_j) E_j   (linear-hat kernel, periodic)
    """
    M      = cells.size
    idx_f  = x / eta - 0.5
    i0     = jnp.floor(idx_f).astype(jnp.int32) % M
    f      = idx_f - jnp.floor(idx_f)
    i1     = (i0 + 1) % M
    return (1.0 - f) * E[i0] + f * E[i1]

@jax.jit
def evaluate_charge_density(x, cells, eta, L, qe=1.0):
    """
    ρ_j = qe * L * ⟨ψ(x − cell_j)⟩   with ψ the same hat kernel.
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
    return qe * L * counts / (x.size * eta)

#%%
"Initialization"
seed = 42

# set physical constants
alpha = 0.1  # Perturbation strength
k = 0.5      # Wave number
dx = 1       # Position dimension
dv = 1       # Velocity dimension

# set number of particles
num_particles = 10_000_000 # 1e8 uses 17 Gb RAM

# Create a mesh
box_length = 2 * jnp.pi / k
num_cells = 10**3
eta = box_length / num_cells
cells = (jnp.arange(num_cells) + 0.5) * eta

# sample initial velocity
key_v, key_x = jr.split(jr.PRNGKey(seed), 2)
v = jr.multivariate_normal(key_v, jnp.zeros(dv), jnp.eye(dv), shape=(num_particles,)).reshape((num_particles, dv))
v = v - jnp.mean(v, axis=0)  # zero-mean velocity

# Sample initial positions with rejection sampling
def spatial_density(x):
    return (1 + alpha * jnp.cos(k * x)) / (2 * jnp.pi / k)
max_value = jnp.max(spatial_density(cells))
domain = (0, box_length)
x = rejection_sample(key_x, spatial_density, domain, max_value = max_value, num_samples=num_particles)

# Compute initial electric field
rho = evaluate_charge_density(x, cells, eta, box_length)
E = jnp.cumsum(rho - 1) * eta 
E = E - jnp.mean(E)

# visualize_initial(x, v[:,0], cells, E, rho, eta, box_length)

#%%
"Forward Euler time stepping"
@jax.jit
def step(x, v, E, cells, eta, dt, box_length):
    E_at_particles = evaluate_field_at_particles(x, cells, E, eta, box_length)
    v_new = v.at[:, 0].add(dt * E_at_particles)
    x_new = jnp.mod(x + dt * v_new[:, 0], box_length)

    E_new = update_electric_field(E, cells, x, v, eta, dt, box_length)
    # rho_new = evaluate_charge_density(x_new, cells, eta, box_length)
    # E_new   = jnp.cumsum(rho_new - 1.0) * eta

    return x_new, v_new, E_new

final_time = 30.0
dt = 0.01
num_steps = int(final_time / dt)
t = 0.
E_L2 = [jnp.sqrt(jnp.sum(E**2) * eta)]

for step_num in tqdm(range(num_steps)):
    x, v, E = step(x, v, E, cells, eta, dt, box_length)
    E = E - jnp.mean(E)  # enforce zero-mean
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
plt.title(f"n={num_particles:.0e}, Δt={dt}, dv={dv}, α={alpha}, C=0, N={num_cells}")
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# %%
"Plot phase space heatmap"

# Downsample for plotting
num_plot = 10_000
key_plot = jr.PRNGKey(123)
idx0 = jr.choice(key_plot, x0.shape[0], shape=(num_plot,), replace=False)
idx = jr.choice(jr.PRNGKey(456), x.shape[0], shape=(num_plot,), replace=False)

# Downsampled particles for plotting
x0_plot = x0[idx0]
v0_plot = v0[idx0]
x_plot = x[idx]
v_plot = v[idx]

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
# Initial phase space KDE
kde1 = sns.kdeplot(x=x0_plot, y=v0_plot[:,0], fill=True, cmap='viridis', ax=axs[0], bw_adjust=0.5, levels=100, thresh=0.05)
axs[0].set_xlabel('Position (x)')
axs[0].set_ylabel('Velocity (v)')
axs[0].set_title('Initial Phase Space Density (KDE), t=0')
cbar1 = plt.colorbar(kde1.get_children()[0], ax=axs[0], label='Density')

# Final phase space KDE
kde2 = sns.kdeplot(x=x_plot, y=v_plot[:,0], fill=True, cmap='viridis', ax=axs[1], bw_adjust=0.5, levels=100, thresh=0.05)
axs[1].set_xlabel('Position (x)')
axs[1].set_ylabel('Velocity (v)')
axs[1].set_title(f'Final Phase Space Density (KDE), t={final_time:.2f}')
cbar2 = plt.colorbar(kde2.get_children()[0], ax=axs[1], label='Density')

plt.tight_layout()
plt.show()


