#%%
import jax.random as jr
import jax.numpy as jnp
from jax import vmap
from tqdm import tqdm

def uniform_sample(domain):
    def proposal_sample(key, num_samples=1):
        return jr.uniform(key, minval=domain[0], maxval=domain[1], shape=(num_samples,))
    return proposal_sample

def uniform_density(domain):
    domain_width = domain[1] - domain[0]
    uniform_density = 1.0 / domain_width
    def proposal_density(x):
        return jnp.where((x >= domain[0]) & (x <= domain[1]), uniform_density, 0.0)
    return proposal_density

def estimate_max_ratio(density_fn, proposal_density, domain):
    x_mesh = jnp.linspace(domain[0], domain[1], 1000)
    target_values = vmap(density_fn)(x_mesh)
    proposal_values = vmap(proposal_density)(x_mesh)
    # Avoid division by zero by adding a small epsilon
    ratios = target_values / (proposal_values + 1e-10)
    estimated_max = jnp.max(ratios)
    return estimated_max

#%%
def rejection_sample(key, density_fn, domain, max_ratio=None, num_samples=1, verbose=False, margin=0.1):
    "sample in parallel"
    
    proposal_sample, proposal_density = uniform_sample(domain), uniform_density(domain)

    # If max_ratio not provided, estimate it from a mesh over the domain
    if max_ratio is None:
        max_ratio = estimate_max_ratio(density_fn, proposal_density, domain)
    
    if verbose:
        print(f"Domain: {domain}")
        print(f"Max ratio: {max_ratio:.2f}")
    
    key, key_propose, key_accept = jr.split(key, 3)
    # sample more than the expected number, by a margin -- 10% by default
    num_candidates = int(num_samples * max_ratio * (1 + margin))
    candidates = proposal_sample(key_propose, num_candidates)
    proposal_values = proposal_density(candidates)
    target_values = density_fn(candidates)
    
    # Accept with probability target/proposal/max_ratio
    accepted = jr.uniform(key_accept, num_candidates) * max_ratio * proposal_values <= target_values
    samples = candidates[accepted]
    
    return samples[:num_samples]
