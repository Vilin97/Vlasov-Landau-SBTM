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

def visualize_initial(x, v, cells, E, rho, eta, L):
    """Visualize initial data."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Histogram of x and desired density
    axs[0].hist(x, bins=50, density=True, alpha=0.6, label='Sampled $x$')
    x_grid = jnp.linspace(0, L, 200)
    axs[0].plot(x_grid, spatial_density(x_grid), 'r-', label='Target density')
    axs[0].plot(cells, rho / L, 'g-', label='$\\rho/L$')
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

#%%
@jax.jit
def evaluate_charge_density(x, cells, eta, w):
    """
    ρ_j = w * Σ_p ψ_eta(X_p − cell_j)   with ψ the hat kernel.
    """
    M      = cells.size                                 # number of cells
    idx_f  = x / eta - 0.5                              # fractional index of particles
    i0     = jnp.floor(idx_f).astype(jnp.int32) % M     # left cell index
    i1     = (i0 + 1) % M                               # right cell index
    f      = idx_f - jnp.floor(idx_f)                   # fractional part
    w0, w1 = 1 - f, f                                   # weights

    counts = (                                          # fractional counts per cell
        jnp.zeros(M)
            .at[i0].add(w0)
            .at[i1].add(w1)
    )
    return w / eta * counts                             # charge density

@jax.jit
def evaluate_field_at_particles(E, x, cells, eta):
    """
    E(x) = Σ_j ψ(x − cell_j) E_j   (linear-hat kernel, periodic)
    """
    M      = cells.size
    idx_f  = x / eta - 0.5
    i0     = jnp.floor(idx_f).astype(jnp.int32) % M
    f      = idx_f - jnp.floor(idx_f)
    i1     = (i0 + 1) % M
    return (1.0 - f) * E[i0] + f * E[i1]

@jax.jit
def update_electric_field(E, x, v, cells, eta, w, dt):
    """
    E_j^{n+1} = E_j^n - dt * w * Σ_i ψ(x_i - cell_j) v_i   (linear-hat kernel, periodic)
    """
    M      = cells.size                                 # number of cells
    idx_f  = x / eta - 0.5                              # fractional index of particles
    i0     = jnp.floor(idx_f).astype(jnp.int32) % M     # left cell index
    i1     = (i0 + 1) % M                               # right cell index
    f      = idx_f - jnp.floor(idx_f)                   # fractional part
    w0, w1 = 1 - f, f                                   # weights

    J = (
        jnp.zeros(M)
          .at[i0].add(w0 * v[:, 0])
          .at[i1].add(w1 * v[:, 0])
    )                                                   # J, before scaling
    dEdt = w / eta * J
    return (E - dt * dEdt).astype(E.dtype)

@jax.jit
def step(x, v, E, cells, eta, dt, box_length):
    "Forward Euler time stepping"
    E_at_particles = evaluate_field_at_particles(E, x, cells, eta)
    v_new = v.at[:, 0].add(dt * E_at_particles)
    x_new = jnp.mod(x + dt * v[:, 0], box_length)
    E_new = update_electric_field(E, x, v, cells, eta, w, dt)
    return x_new, v_new, E_new

# %%
seed = 42

# set physical constants
q = 1       # particle charge
dx = 1       # Position dimension
dv = 1       # Velocity dimension

alpha = 0.1  # Perturbation strength
k = 0.5      # Wave number
L = 2 * jnp.pi / k # domain size

for n in [10**6, 10**7, 10**8]:
    for M in [1000, 100, 10]:
        for dt in [0.1, 0.01, 0.001]:
            print(f"Running n={n:.0e}, M={M}, dt={dt}")

            # set numerical constants
            eta = L / M
            cells = (jnp.arange(M) + 0.5) * eta
            w = q*L/n    # particle weight / charge

            # sample initial velocity
            key_v, key_x = jr.split(jr.PRNGKey(seed), 2)
            v = jr.multivariate_normal(key_v, jnp.zeros(dv), jnp.eye(dv), shape=(n,)).reshape((n, dv))
            v = v - jnp.mean(v, axis=0)  # zero-mean velocity

            # Sample initial positions with rejection sampling
            def spatial_density(x):
                return (1 + alpha * jnp.cos(k * x)) / (2 * jnp.pi / k)
            max_value = jnp.max(spatial_density(cells))
            domain = (0, L)
            x = rejection_sample(key_x, spatial_density, domain, max_value = max_value, num_samples=n)

            # Compute initial electric field
            rho = evaluate_charge_density(x, cells, eta, w)
            E = jnp.cumsum(rho - 1) * eta 
            E = E - jnp.mean(E)
            visualize_initial(x, v[:,0], cells, E, rho, eta, L)

            final_time = 30.0
            num_steps = int(final_time / dt)
            t = 0.
            E_L2 = [jnp.sqrt(jnp.sum(E**2) * eta)]

            for step_num in tqdm(range(num_steps)):
                x, v, E = step(x, v, E, cells, eta, dt, L)
                E = E - jnp.mean(E)  # enforce zero-mean
                t += dt
                E_L2.append(jnp.sqrt(jnp.sum(E**2) * eta))

            # plot L2 norm of E over time
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

            maxima_indices = argrelextrema(n_mask, np.greater, order=5)[0]
            mt = t_mask[maxima_indices]
            mv = n_mask[maxima_indices]
            plt.scatter(mt, mv, color='g', marker='o', zorder=5)
            coeffs = np.polyfit(mt, np.log(mv), 1)
            fit = np.exp(coeffs[1] + coeffs[0] * t_mask)
            plt.plot(t_mask, fit, 'g--', label=fr'$e^{{\beta t}}, \beta={coeffs[0]:.3f}$')

            plt.xlabel('Time')
            plt.ylabel(r'$||E||_{L^2}$')
            plt.title(f"n={n:.0e}, Δt={dt}, dv={dv}, α={alpha}, C=0, M={M}")
            plt.yscale('log')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            plt.savefig(f"data/plots/electric_field_norm/collisionless_1d_1v/landau_damping_n{n:.0e}_M{M}_dt{dt}.png")
            plt.show()