#%%
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
from src import density as density_module

import importlib
importlib.reload(density_module)

# Set parameters
alpha = 0.4  # Perturbation strength
k = 0.5       # Wave number
dx = 1        # Spatial dimension
dv = 2        # Velocity dimension
sample_size = 1000  # Number of samples to generate

# Initialize random key
key = jrandom.PRNGKey(42)

# Initialize CosineNormal density
density = density_module.CosineNormal(alpha=alpha, k=k, dx=dx, dv=dv)

print(f"Initialized CosineNormal with alpha={alpha}, k={k}, dx={dx}, dv={dv}")
print(f"Domain x: {density.domain_x}")

#%%
# Sample from the distribution
print(f"Generating {sample_size} samples...")
x_samples, v_samples = density.sample(key, size=sample_size)

#%%
# Evaluate density at sampled points
density_values = density(x_samples, v_samples)
print(f"Density values -- min: {jnp.min(density_values)}, max: {jnp.max(density_values)}, mean: {jnp.mean(density_values)}")

#%%
# Calculate score at sampled points
score_values = density.score(x_samples, v_samples)
print(f"Score shape: {score_values.shape}")
print(f"Score statistics -- min: {jnp.min(score_values)}, max: {jnp.max(score_values)}, mean: {jnp.mean(score_values)}")

#%%
# Visualize the samples and density
plt.figure(figsize=(15, 10))

# Plot 1: Spatial distribution of samples
plt.subplot(2, 2, 1)
hist_x, bins_x, _ = plt.hist(np.array(x_samples), bins=30, density=True, alpha=0.7, label='Samples')
# Overlay theoretical marginal density
x_points = np.linspace(density.domain_x[0], density.domain_x[1], 200)
density_x = np.array([density.density_x(x) for x in x_points])
plt.plot(x_points, density_x, 'r-', linewidth=2, label='Theoretical')
plt.title('Spatial Distribution of Samples')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()

# Plot 2: Velocity distribution for first component
plt.subplot(2, 2, 2)
hist_v, bins_v, _ = plt.hist(np.array(v_samples[:, 0]), bins=30, density=True, alpha=0.7, label='Samples')
# Overlay theoretical marginal density
v_points = np.linspace(np.min(v_samples[:, 0]), np.max(v_samples[:, 0]), 200)
density_v = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * v_points**2)
plt.plot(v_points, density_v, 'r-', linewidth=2, label='Theoretical')
plt.title('Velocity Distribution (1st component)')
plt.xlabel('v₁')
plt.ylabel('Density')
plt.legend()

# Plot 3: 2D scatter of position and first velocity component
plt.subplot(2, 2, 3)
plt.scatter(np.array(x_samples), np.array(v_samples[:, 0]), alpha=0.5, s=5)
plt.title('Position vs Velocity (1st component)')
plt.xlabel('x')
plt.ylabel('v₁')

# Plot 4: Theoretical density function over the domain
# Create a 2D slice of the density function (using first velocity component)
x_grid = np.linspace(density.domain_x[0], density.domain_x[1], 100)
v_grid = np.linspace(-3, 3, 100)
X, V = np.meshgrid(x_grid, v_grid)
positions = np.vstack([X.ravel(), V.ravel()]).T

# Convert to JAX arrays
positions_x = jnp.array(positions[:, 0]).reshape(-1, 1)
# Create velocity array with appropriate dimensions
positions_v = jnp.zeros((positions.shape[0], dv))
positions_v = positions_v.at[:, 0].set(positions[:, 1])  # Set first velocity component

# Evaluate density
Z = density(positions_x, positions_v)
Z = np.array(Z).reshape(X.shape)

plt.subplot(2, 2, 4)
plt.contourf(X, V, Z, cmap='viridis')
plt.colorbar(label='Density')
plt.title('Theoretical Density (v₁ slice)')
plt.xlabel('x')
plt.ylabel('v₁')

plt.tight_layout()
plt.savefig('../plots/cosine_normal_density.png')
plt.show()

print("Script completed. Visualization saved as 'cosine_normal_density.png'")

# %%
