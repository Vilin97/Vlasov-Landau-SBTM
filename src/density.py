from jax import grad, vmap
from jax.scipy.stats import multivariate_normal
import jax.random as jrandom
import jax.numpy as jnp
from flax import nnx

@nnx.jit(static_argnames='log_density')
def score_log_density(log_density, x, v):
    return vmap(lambda x, v: grad(lambda v_: log_density(x, v_))(v))(x, v)

class Density:
    def __init__(self):
        pass

    def log_density(self, x, v):
        """Compute the log_density of the distribution at (x,v)."""
        log_density = jnp.log(jnp.clip(self.density(x, v), a_min=1e-10))
        return log_density
    
    def __call__(self, x, v):
        """Alias for the density method."""
        return self.density(x, v)

    def score(self, x, v):
        """Compute the score (gradient of log_density with respect to v)."""
        return score_log_density(self.log_density, x, v)

    def density(self, x, v):
        """Compute the density of the distribution at (x,v)."""
        raise NotImplementedError("Subclasses must implement the density method.")

    def sample(self, key, size=1):
        """Generate samples from the distribution."""
        raise NotImplementedError("Subclasses must implement the sample method.")

class CosineNormal(Density):
    def __init__(self, alpha=0.01, k=0.5):
        """
        alpha: Perturbation strength, typically small (default: 0.1)
        k: Wave number (default: 0.5)
        """
        super().__init__()
        self.alpha = alpha
        self.k = k
        domain_x = jnp.array([0, 2 * jnp.pi/k])
        self.domain_x = domain_x
        self.domain_size_x = domain_x[1] - domain_x[0]
        
    def log_density(self, x, v):
        """
        Compute log of f₀(x,v) = log[(1 + α cos(kx))/(2π) * exp(-|v|²/2)]
        = log(1 + α cos(kx)) - log(2π) - |v|²/2
        """
        spatial_part = jnp.log(jnp.clip(1 + self.alpha * jnp.cos(self.k * x), a_min=1e-10))
        velocity_part = -jnp.sum(v**2, axis=-1) / 2
        log_2pi = jnp.log(2 * jnp.pi)
        
        return spatial_part - log_2pi + velocity_part
    
    def density(self, x, v):
        """
        f₀(x,v) = (1 + α cos(kx))/(2π) * exp(-|v|²/2)
        """
        spatial_part = (1 + self.alpha * jnp.cos(self.k * x)) / (2 * jnp.pi)
        velocity_part = jnp.exp(-jnp.sum(v**2, axis=-1) / 2)
        return spatial_part * velocity_part
    
    def sample(self, key, size=1):
        """
        Returns: Tuple of (x, v) samples
        """
        key_x, key_v = jrandom.split(key)
        
        # Sample velocity directly from normal distribution
        v_samples = jrandom.normal(key_v, shape=(size, 1))
        
        # Maximum value of spatial density for rejection sampling
        max_density = (1 + self.alpha) / (2 * jnp.pi)
        
        # Define the spatial density function
        def spatial_density(x):
            return (1 + self.alpha * jnp.cos(self.k * x)) / (2 * jnp.pi)
        
        # Generate all samples
        x_keys = jrandom.split(key_x, size)
        x_samples = vmap(lambda k: rejection_sample(k, spatial_density, self.domain_x, max_density))(x_keys)
        
        return x_samples, v_samples
