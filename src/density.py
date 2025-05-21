from jax import grad, vmap
from jax.scipy.stats import multivariate_normal
import jax.random as jr
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
        if x.shape != (1,):
            raise NotImplementedError("log_density does not accept a batch of data")
        
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
        key_x, key_v = jr.split(key)
        
        # Sample velocity directly from multivariate normal distribution
        v_samples = jr.multivariate_normal(key_v, self.mean, self.cov, shape=(size,))
        
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

class TwoStream(Density):
    r"""
    f₀(x,v)= (1+α cos(kx))/(2π) · [e^{-(v₁-c)²/2}+e^{-(v₁+c)²/2}] · ∏_{j≥2} e^{-vⱼ²/2}.
    """
    def __init__(self, *, alpha=1/200, k=1/5, c=2.4, dx=1, dv=2):
        super().__init__()
        if dx != 1:
            raise NotImplementedError("dx must be 1")
        self.alpha, self.k, self.c = alpha, k, c
        self.dv = dv
        self.domain_x = jnp.array([0.0, 2 * jnp.pi / self.k])   # periodic domain

    # ── PDF ──────────────────────────────────────────────────────────
    def density(self, x, v):
        v1, v_rest = v[..., 0], v[..., 1:]
        spatial = (1 + self.alpha * jnp.cos(self.k * x[..., 0])) / (2 * jnp.pi)
        longi   = jnp.exp(-(v1 - self.c) ** 2 / 2) + jnp.exp(-(v1 + self.c) ** 2 / 2)
        transv  = jnp.exp(-jnp.sum(v_rest ** 2, axis=-1) / 2)
        return spatial * longi * transv

    # ── sampling ─────────────────────────────────────────────────────
    def sample(self, key, size):
        kx, kv = jr.split(key)
        # x  — rejection sampling on spatial part
        def g(u):  # un-normalised density on x
            return (1 + self.alpha * jnp.cos(self.k * u)) / (2 * jnp.pi)
        x = rejection_sample(kx, g, self.domain_x, num_samples=size)[:, None]

        # v₁ — equal-weight mixture ½𝒩(c,1)+½𝒩(−c,1)
        kv1, kvrest, ksign = jr.split(kv, 3)
        sign = jr.choice(ksign, jnp.array([1.0, -1.0]), shape=(size, 1))
        v1   = sign * self.c + jr.normal(kv1, shape=(size, 1))

        # v₂,…,v_{dv}  — i.i.d. 𝒩(0,1)
        v_rest = jr.normal(kvrest, shape=(size, self.dv - 1))

        v = jnp.concatenate([v1, v_rest], axis=1)
        return x, v