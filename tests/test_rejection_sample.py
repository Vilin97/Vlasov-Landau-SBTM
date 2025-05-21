import unittest
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import argparse
import sys

from src.rejection_sample import rejection_sample

class TestRejectionSample(unittest.TestCase):
    verbose = False  # Class attribute to store verbosity setting
    
    def setUp(self):
        # Print test name if verbose is enabled
        if TestRejectionSample.verbose:
            print(f"\nRunning test: {self._testMethodName}")
    
    def test_uniform_sampling(self):
        """Test rejection sampling from a uniform distribution."""
        key = jr.PRNGKey(42)
        domain = jnp.array([0.0, 1.0])
        
        # Generate samples 
        n_samples = 1000
        samples = rejection_sample(key, lambda x: 1/(domain[1] - domain[0]), domain, num_samples=n_samples, verbose=self.verbose)
        
        # Check shape of returned samples
        self.assertEqual(samples.shape, (n_samples,))
        
        # Test the samples are within domain
        self.assertTrue(jnp.all(samples >= domain[0]))
        self.assertTrue(jnp.all(samples <= domain[1]))
        
        # Test uniformity using Kolmogorov-Smirnov test
        uniform_dist = stats.uniform(domain[0], domain[1] - domain[0])
        ks_stat, p_value = stats.kstest(np.array(samples), uniform_dist.cdf)
        self.assertTrue(p_value > 0.05, f"KS test failed with p-value {p_value}")
    
    def test_gaussian_sampling(self):
        """Test rejection sampling from a Gaussian distribution."""
        key = jr.PRNGKey(42)
        mu, sigma = 0.0, 1.0
        
        # Gaussian density function
        def gaussian_density(x):
            return jnp.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * jnp.sqrt(2 * jnp.pi))
        
        # 6-sigma domain for the Gaussian
        domain = jnp.array([mu - 6*sigma, mu + 6*sigma])
        
        # Generate samples
        n_samples = 1000
        samples = rejection_sample(key, gaussian_density, domain, num_samples=n_samples, verbose=self.verbose)
        
        # Test samples against a normal distribution using Kolmogorov-Smirnov test
        normal_dist = stats.norm(mu, sigma)
        ks_stat, p_value = stats.kstest(np.array(samples), normal_dist.cdf)
        self.assertTrue(p_value > 0.05, f"KS test failed with p-value {p_value}")
    
    def test_with_known_max_ratio(self):
        """Test rejection sampling with a provided max_ratio value."""
        key = jr.PRNGKey(42)
        
        # Simple triangular density function
        def triangular_density(x):
            return 2.0 * (1.0 - jnp.abs(x))
        
        domain = jnp.array([-1.0, 1.0])
        max_ratio = 4 
        
        # Generate samples 
        n_samples = 1000
        samples = rejection_sample(key, triangular_density, domain, max_ratio=max_ratio, num_samples=n_samples, verbose=self.verbose)
        
        # Test the samples are within domain
        self.assertTrue(jnp.all(samples >= domain[0]))
        self.assertTrue(jnp.all(samples <= domain[1]))
        
        # Triangular distribution parameters for scipy
        tri_dist = stats.triang(c=0.5, loc=-1.0, scale=2.0)
        ks_stat, p_value = stats.kstest(np.array(samples), tri_dist.cdf)
        self.assertTrue(p_value > 0.05, f"KS test failed with p-value {p_value}")
    
    def test_with_nonuniform_proposal(self):
        """Test rejection sampling using a non-uniform proposal distribution."""
        key = jr.PRNGKey(42)
        
        # Target: Standard Gaussian
        mu_target, sigma_target = 0.0, 1.0
        def target_density(x):
            return jnp.exp(-0.5 * ((x - mu_target) / sigma_target) ** 2) / (sigma_target * jnp.sqrt(2 * jnp.pi))
        
        # Proposal: Wider Gaussian (more variance)
        mu_proposal, sigma_proposal = 0.0, 1.5
        def proposal_density(x):
            return jnp.exp(-0.5 * ((x - mu_proposal) / sigma_proposal) ** 2) / (sigma_proposal * jnp.sqrt(2 * jnp.pi))
        
        domain = jnp.array([-6.0, 6.0])
        n_samples = 1000
        
        # Create proposal sampling function that draws from the Gaussian proposal
        def proposal_sample(key):
            return mu_proposal + sigma_proposal * jr.normal(key)
        
        # Bundle proposal as (sample_fn, density_fn)
        proposal = (proposal_sample, proposal_density)
        
        # Generate samples using the non-uniform proposal
        samples = rejection_sample(
            key, 
            target_density, 
            domain, 
            proposal=proposal,
            num_samples=n_samples, 
            verbose=self.verbose
        )
        
        # Test samples against the target normal distribution
        normal_dist = stats.norm(mu_target, sigma_target)
        ks_stat, p_value = stats.kstest(np.array(samples), normal_dist.cdf)
        self.assertTrue(p_value > 0.05, f"KS test failed with p-value {p_value}")
    
    def test_single_sample(self):
        """Test that when num_samples=1, a single sample is returned."""
        key = jr.PRNGKey(42)
        
        def uniform_density(x):
            return jnp.ones_like(x)
        
        domain = jnp.array([0.0, 1.0])
        
        # Generate a single sample
        sample = rejection_sample(key, uniform_density, domain, num_samples=1, verbose=self.verbose)
        
        # Check that it's a single-element array with shape (1,)
        self.assertEqual(sample.shape, (1,))
        
        # Test the sample is within domain
        self.assertTrue(sample >= domain[0] and sample <= domain[1])

def visualize_samples():
    """
    Visualize the samples (not a test, but useful for debugging).
    Run this separately, not as part of automated testing.
    """
    key = jr.PRNGKey(42)
    
    # Gaussian density function
    mu, sigma = 0.0, 1.0
    def gaussian_density(x):
        return jnp.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * jnp.sqrt(2 * jnp.pi))
    
    domain = jnp.array([mu - 4*sigma, mu + 4*sigma])
    
    # Generate samples 
    n_samples = 5000
    samples = rejection_sample(key, gaussian_density, domain, num_samples=n_samples, verbose=self.verbose)
    
    # Plot histogram and theoretical PDF
    plt.figure(figsize=(10, 6))
    plt.hist(np.array(samples), bins=50, density=True, alpha=0.6, label='Rejection Samples')
    
    x = np.linspace(domain[0], domain[1], 1000)
    y = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, y, 'r-', lw=2, label='Theoretical PDF')
    
    plt.title('Rejection Sampling of Gaussian Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.savefig('rejection_sampling_test.png')
    plt.close()

if __name__ == '__main__':
    # Change this to True to enable verbose output
    TestRejectionSample.verbose = False
    
    # Run the tests
    unittest.main()
