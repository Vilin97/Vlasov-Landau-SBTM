from jax import grad, vmap
from jax.scipy.stats import multivariate_normal
import jax.random as jrandom
import jax.numpy as jnp
from flax import nnx
from src.rejection_sample import rejection_sample

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
    
    def __call__(self, X, V):
        """
        Compute the density for batched inputs X and V.
        """
        return vmap(self.density)(X, V)

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
    def __init__(self, alpha=0.1, k=0.5, dx=1, dv=2):
        """
        f(x,v) = (1 + α cos(kx)) * N(v|0,I)

        alpha: Perturbation strength, typically small (default: 0.1)
        k: Wave number (default: 0.5)
        dx: Dimension of spatial coordinates (must be 1)
        dv: Dimension of velocity coordinates
        """
        super().__init__()
        self.alpha = alpha
        self.k = k
        self.dx = dx
        self.dv = dv
        
        if dx != 1:
            raise NotImplementedError("Only 1D spatial dimension is supported")
            
        domain_x = jnp.array([0, 2 * jnp.pi/k])
        self.domain_x = domain_x
        self.domain_size_x = domain_x[1] - domain_x[0]
        
        # Zero mean and identity covariance for velocity distribution
        self.mean = jnp.zeros(dv)
        self.cov = jnp.eye(dv)
        
    def log_density(self, x, v):
        spatial_part = jnp.log(jnp.clip(1 + self.alpha * jnp.cos(self.k * x), a_min=1e-10))
        velocity_part = multivariate_normal.logpdf(v, self.mean, self.cov)
        
        return (spatial_part + velocity_part)[0]
    
    def density(self, x, v):
        return jnp.exp(self.log_density(x, v))
    
    def density_x(self, x):
        """
        Spatial part of the density function: (1 + α cos(kx)) / self.domain_size_x
        """
        return (1 + self.alpha * jnp.cos(self.k * x)) / self.domain_size_x
    
    def density_v(self, v):
        """
        Velocity part of the density function: N(v | 0, I)
        """
        return multivariate_normal.pdf(v, self.mean, self.cov)
    
    def sample(self, key, size=1):
        """
        Returns: Tuple of (x, v) samples
        """
        key_x, key_v = jrandom.split(key)
        
        # Sample velocity directly from multivariate normal distribution
        v_samples = jrandom.multivariate_normal(key_v, self.mean, self.cov, shape=(size,))
        
        # Define the spatial density function
        def spatial_density(x):
            return (1 + self.alpha * jnp.cos(self.k * x)) / (2 * jnp.pi)
        
        # Generate samples using rejection sampling
        x_samples = rejection_sample(
            key_x, 
            spatial_density, 
            self.domain_x, 
            num_samples=size
        )
        
        x_samples = x_samples.reshape(size, self.dx)
        
        return x_samples, v_samples
