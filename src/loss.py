import jax
import jax.numpy as jnp
from flax import nnx
from typing import Callable

def divergence_wrt_v(f: Callable, mode: str, n: int = 100):
    """
    Compute the divergence of a vector field f with respect to v, where f(x,v) -> output with shape of v.
    
    Args:
        f: Callable
            Function of form f(x,v) that returns output with same shape as v.
        mode: str
            Mode of divergence computation.
        n: int
            Number of samples for stochastic estimation.
    
    Returns:
        Callable that computes the divergence at a point (x,v) with respect to v.
    """
    # gaussian and rademacher are ~30% faster than exact methods and denoised
    assert mode in ['forward', 'reverse', 'approximate_gaussian', 'approximate_rademacher', 'denoised'], "Invalid mode"
    
    # Create a wrapper that treats only v as the variable for differentiation
    def f_wrapper(v, x):
        return f(x, v)
    
    if mode == 'forward':
        @jax.jit
        def div(x, v):
            return jnp.trace(jax.jacfwd(f_wrapper, argnums=0)(v, x))
        return div
        
    if mode == 'reverse':
        @jax.jit
        def div(x, v):
            return jnp.trace(jax.jacrev(f_wrapper, argnums=0)(v, x))
        return div
        
    if mode == 'denoised':
        alpha = jnp.float32(0.1)
        @jax.jit
        def div(x, v, key):
            def denoise(key):
                epsilon = jax.random.normal(key, v.shape, dtype=v.dtype)
                return jnp.sum(
                    (f(x, v + alpha * epsilon) - f(x, v - alpha * epsilon)) * epsilon
                ) / (2*alpha)
            return jax.vmap(denoise)(jax.random.split(key, n)).mean()
        return div
    else:
        @jax.jit
        def div(x, v, key):
            def vJv(key):
                # Define a partial function that fixes x
                fixed_x_f = lambda v_: f(x, v_)
                # Get vector-Jacobian product function
                _, vjp_fun = jax.vjp(fixed_x_f, v)
                # Generate random vector
                rand_gen = jax.random.normal if mode == 'approximate_gaussian' else jax.random.rademacher
                epsilon = rand_gen(key, v.shape, dtype=v.dtype)
                # Compute vᵀ(∂f/∂v)ᵀv = vᵀJ_v[f]ᵀv
                return jnp.sum(vjp_fun(epsilon)[0] * epsilon)
            return jax.vmap(vJv)(jax.random.split(key, n)).mean()
        return div

@nnx.jit
def explicit_score_matching_loss(s, x_batch, v_batch, target_score_values):
    """
    Compute the score matching loss between the score function s and the true score.
    1/n ∑ᵢ ||s(xᵢ,vᵢ) - ∇ᵥlog p(xᵢ,vᵢ)||²
    
    Args:
        s: Score model function that takes (x,v) and returns score
        x_batch: Position batch of shape (batch_size, dx)
        v_batch: Velocity batch of shape (batch_size, dv)
        target_score_values: True score values of shape (batch_size, dv)
    
    Returns:
        Mean squared error loss
    """
    # Compute predictions for all samples in the batch
    predicted_scores = jax.vmap(s)(x_batch, v_batch)
    
    # Compute mean squared error
    return jnp.mean(jnp.sum(jnp.square(predicted_scores - target_score_values), axis=-1))

@nnx.jit
def weighted_explicit_score_matching_loss(s, x_batch, v_batch, target_score_values, weighting):
    """
    Compute the weighted score matching loss between the score function s and the true score.
    1/n ∑ᵢ ⟨s(xᵢ,vᵢ) - ∇ᵥlog p(xᵢ,vᵢ), D[i] (s(xᵢ,vᵢ) - ∇ᵥlog p(xᵢ,vᵢ))⟩
    
    Args:
        s: Score model function that takes (x,v) and returns score
        x_batch: Position batch of shape (batch_size, dx)
        v_batch: Velocity batch of shape (batch_size, dv)
        target_score_values: True score values of shape (batch_size, dv)
        weighting: Weighting matrices of shape (batch_size, dv, dv) or (dv, dv)
    
    Returns:
        Weighted mean squared error loss
    """
    def weighted_loss(x, v, target, D):
        pred = s(x, v)
        diff = pred - target
        # If D is (dv, dv), dot product is diff^T @ D @ diff
        # If D is a scalar, it's equivalent to diff^T @ D*I @ diff = D * (diff^T @ diff)
        if D.ndim == 2:
            return jnp.dot(diff, jnp.dot(D, diff))
        else:
            return jnp.dot(diff, diff) * D
    
    return jnp.mean(jax.vmap(weighted_loss)(x_batch, v_batch, target_score_values, weighting))

@nnx.jit(static_argnames=['div_mode', 'n_samples'])
def implicit_score_matching_loss(s, x_batch, v_batch, key=None, div_mode='reverse', n_samples=100):
    """
    Compute the implicit score matching loss for score function s(x,v)
    1/n ∑ᵢ ||s(xᵢ,vᵢ)||^2 + 2 ∇ᵥ⋅s(xᵢ,vᵢ)
    
    Args:
        s: Score model function that takes (x,v) and returns score
        x_batch: Position batch of shape (batch_size, dx)
        v_batch: Velocity batch of shape (batch_size, dv)
        key: Optional JAX PRNGKey for stochastic estimators
        div_mode: Mode for divergence computation: 'forward', 'reverse', 'approximate_gaussian', 'approximate_rademacher', 'denoised'
        n_samples: Number of samples for stochastic divergence estimation
    
    Returns:
        Implicit score matching loss
    """
    # Get the appropriate divergence function
    div_fn = divergence_wrt_v(s, div_mode, n_samples)
    
    def compute_loss(x, v, key=None):
        # Compute squared norm of score
        score = s(x, v)
        squared_norm = jnp.sum(jnp.square(score))
        
        # Compute divergence based on mode
        if div_mode in ['approximate_gaussian', 'approximate_rademacher', 'denoised']:
            assert key is not None, "For stochastic divergence estimation, key must be provided"
            div = div_fn(x, v, key) if key is not None else div_fn(x, v)
        else:
            div = div_fn(x, v)
            
        return squared_norm + 2 * div
    
    if div_mode in ['forward', 'reverse']:
        # For exact methods, we can directly vmap over the batch
        return jnp.mean(jax.vmap(compute_loss)(x_batch, v_batch))
    else:
        # For stochastic methods, we need to handle the random keys
        batch_size = x_batch.shape[0]
        keys = jax.random.split(key, batch_size)
        loss_fn = lambda x, v, k: compute_loss(x, v, k)
        return jnp.mean(jax.vmap(loss_fn)(x_batch, v_batch, keys))
