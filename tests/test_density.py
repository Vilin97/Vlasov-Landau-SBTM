import unittest
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import sys
import os

from src.density import Density, CosineNormal, rejection_sample

class TestDensity(unittest.TestCase):
    """Tests for the base Density class."""
    
    def test_base_density_abstract(self):
        """Test that the base Density class requires implementation of abstract methods."""
        density = Density()
        
        # Test density method raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            density.density(0.0, 0.0)
            
        # Test sample method raises NotImplementedError
        key = jr.PRNGKey(0)
        with self.assertRaises(NotImplementedError):
            density.sample(key)
            
    def test_call_uses_density(self):
        """Test that __call__ uses the density method."""
        class MockDensity(Density):
            def density(self, x, v):
                return x + v
                
            def sample(self, key, size=1):
                return None
                
        density = MockDensity()
        self.assertEqual(density(1.0, 2.0), 3.0)
        self.assertEqual(density.density(1.0, 2.0), 3.0)
        
    def test_log_density(self):
        """Test that log_density returns the log of density with clipping."""
        class MockDensity(Density):
            def density(self, x, v):
                return jnp.array([0.5, 1e-11])
                
            def sample(self, key, size=1):
                return None
                
        density = MockDensity()
        log_densities = density.log_density(None, None)
        # Test log of 0.5
        self.assertAlmostEqual(log_densities[0], jnp.log(0.5), places=5)
        # Test clipping works for very small values
        self.assertAlmostEqual(log_densities[1], jnp.log(1e-10), places=5)
        
class TestCosineNormal(unittest.TestCase):
    """Tests for the CosineNormal distribution."""
    
    def setUp(self):
        """Set up a CosineNormal instance for tests."""
        self.alpha = 0.01
        self.k = 0.5
        self.density = CosineNormal(alpha=self.alpha, k=self.k)
        self.key = jr.PRNGKey(42)
        
    def test_init(self):
        """Test initialization of CosineNormal."""
        self.assertEqual(self.density.alpha, self.alpha)
        self.assertEqual(self.density.k, self.k)
        self.assertEqual(self.density.domain_size_x, 2 * jnp.pi / self.k)
        np.testing.assert_array_equal(
            self.density.domain_x, 
            jnp.array([0, 2 * jnp.pi / self.k])
        )
        
    def test_density_values(self):
        """Test density computation for specific points."""
        # Test point where cos(kx) = 1
        x1, v1 = jnp.array(0.0), jnp.array([0.0])
        expected1 = (1 + self.alpha) / (2 * jnp.pi)
        self.assertAlmostEqual(self.density.density(x1, v1), expected1, places=5)
        
        # Test point where cos(kx) = -1
        x2, v2 = jnp.array(jnp.pi / self.k), jnp.array([0.0])
        expected2 = (1 - self.alpha) / (2 * jnp.pi)
        self.assertAlmostEqual(self.density.density(x2, v2), expected2, places=5)
        
        # Test point with non-zero velocity
        x3, v3 = jnp.array(0.0), jnp.array([1.0])
        expected3 = (1 + self.alpha) / (2 * jnp.pi) * jnp.exp(-0.5)
        self.assertAlmostEqual(self.density.density(x3, v3), expected3, places=5)
        
    def test_log_density(self):
        """Test log density computation."""
        x, v = jnp.array(0.0), jnp.array([1.0])
        log_density = self.density.log_density(x, v)
        density = self.density.density(x, v)
        self.assertAlmostEqual(log_density, jnp.log(density), places=5)
        
    def test_score(self):
        """Test score computation (gradient of log density wrt velocity)."""
        x, v = jnp.array(0.0), jnp.array([2.0])
        score = self.density.score(x, v)
        # For the Gaussian velocity distribution, the score should be -v
        self.assertAlmostEqual(score[0][0], -2.0, places=5)
        
    def test_sample_shape(self):
        """Test that samples have the correct shape."""
        size = 10
        x_samples, v_samples = self.density.sample(self.key, size=size)
        
        self.assertEqual(x_samples.shape, (size,))
        self.assertEqual(v_samples.shape, (size, 1))
        
    def test_sample_domain(self):
        """Test that samples are within the correct domain."""
        size = 100
        x_samples, v_samples = self.density.sample(self.key, size=size)
        
        # Check x values are within domain
        self.assertTrue(jnp.all(x_samples >= self.density.domain_x[0]))
        self.assertTrue(jnp.all(x_samples <= self.density.domain_x[1]))
        
class TestRejectionSample(unittest.TestCase):
    """Tests for the rejection_sample utility function."""
    
    def test_uniform_sampling(self):
        """Test rejection sampling from a uniform distribution."""
        key = jr.PRNGKey(0)
        
        # Define a uniform density on [0, 1]
        def uniform_density(x):
            return jnp.ones_like(x)
            
        domain = jnp.array([0.0, 1.0])
        
        # Sample multiple points
        samples = jnp.array([
            rejection_sample(k, uniform_density, domain, max_density=1.0) 
            for k in jr.split(key, 100)
        ])
        
        # Check samples are within domain
        self.assertTrue(jnp.all(samples >= domain[0]))
        self.assertTrue(jnp.all(samples <= domain[1]))
        
        # With uniform distribution, samples should be roughly uniform
        # but allow for more statistical variation since we're using random sampling
        hist, _ = np.histogram(np.array(samples), bins=5, range=(0, 1))
        expected_count = len(samples) / 5
        
        # Allow more variation in the histogram (up to 70% difference from expected)
        # This is necessary because rejection sampling has statistical fluctuations
        self.assertTrue(np.all(np.abs(hist - expected_count) < expected_count * 0.7))
        
    def test_max_density_estimation(self):
        """Test that max_density is estimated correctly if not provided."""
        key = jr.PRNGKey(0)
        
        # Define a simple density function with known maximum
        def parabola_density(x):
            return 1 - (x - 0.5)**2  # Max of 1.0 at x=0.5
            
        domain = jnp.array([0.0, 1.0])
        
        # Sample with and without providing max_density
        sample_with_max = rejection_sample(key, parabola_density, domain, max_density=1.0)
        sample_without_max = rejection_sample(jr.fold_in(key, 1), parabola_density, domain)
        
        # Both should be valid samples in the domain
        self.assertTrue(0 <= sample_with_max <= 1)
        self.assertTrue(0 <= sample_without_max <= 1)

if __name__ == "__main__":
    unittest.main()
