import jax
import jax.random as jrandom
import jax.numpy as jnp
from jax import vmap
from tqdm import tqdm

def rejection_sample(key, density_fn, domain, proposal=None, max_ratio=None, num_samples=1, verbose=False):
    """
    Perform rejection sampling from a given density function using a proposal distribution.
    
    Args:
        key: JAX PRNGKey
        density_fn: Function that computes target density at a point
        domain: Array of [min, max] bounds for sampling
        proposal_fn: Function that generates samples and computes proposal density
                    If None, uses uniform distribution over domain
        max_ratio: Maximum ratio of target/proposal densities
                  If None, estimated from the densities
        num_samples: Number of samples to generate (default: 1)
        verbose: Whether to print statistics about rejection sampling efficiency (default: False)
    
    Returns:
        An array of samples from the distribution
    """
    # If proposal_fn not provided, use uniform distribution over domain
    if proposal is None:
        domain_width = domain[1] - domain[0]
        uniform_density = 1.0 / domain_width
        
        def proposal_sample(key):
            return jrandom.uniform(key, minval=domain[0], maxval=domain[1])
        
        def proposal_density(x):
            return jnp.where((x >= domain[0]) & (x <= domain[1]), uniform_density, 0.0)
    else:
        proposal_sample, proposal_density = proposal
    
    # If max_ratio not provided, estimate it from a mesh over the domain
    if max_ratio is None:
        x_mesh = jnp.linspace(domain[0], domain[1], 1000)
        target_values = vmap(density_fn)(x_mesh)
        proposal_values = vmap(proposal_density)(x_mesh)
        # Avoid division by zero by adding a small epsilon
        ratios = target_values / (proposal_values + 1e-10)
        estimated_max = jnp.max(ratios)
        # Add 20% safety margin
        max_ratio = 1.2 * estimated_max
    
    if verbose:
        print(f"Domain: {domain}")
        print(f"Max ratio: {max_ratio:.2f}")
    
    samples = []
    total_steps = 0
    
    for i in tqdm(range(num_samples), desc="Rejection sampling", disable=not verbose):
        accepted = False
        x = jnp.zeros(())
        
        while not accepted:
            key, key_x, key_u = jrandom.split(key, 3)
            
            # Sample from proposal distribution
            x_candidate = proposal_sample(key_x)
            proposal_value = proposal_density(x_candidate)
            target_value = density_fn(x_candidate)
            
            # Compute acceptance ratio
            ratio = target_value / (proposal_value + 1e-10)  # Avoid division by zero
            
            # Assert that the max_ratio is adequate
            assert max_ratio >= ratio, f"Max ratio {max_ratio} too small for ratio {ratio} at x={x_candidate} (target={target_value}, proposal={proposal_value})"
            
            # Accept with probability target/proposal/max_ratio
            accepted = jrandom.uniform(key_u) * max_ratio * proposal_value <= target_value
            
            total_steps += 1
            
            # Update x if accepted
            if accepted:
                x = x_candidate
        
        samples.append(x)
    
    if verbose:
        expected_steps = max_ratio
        actual_steps = total_steps / num_samples
        print(f"Max ratio: {max_ratio:.2f}")
        print(f"Expected avg rejection steps: {expected_steps:.2f}")
        print(f"Actual avg rejection steps: {actual_steps:.2f}")
    
    return jnp.array(samples)
