import unittest
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mesh import Mesh

class TestMesh(unittest.TestCase):
    def test_init_1d(self):
        """Test 1D mesh initialization."""
        mesh = Mesh(10.0, 20)
        self.assertEqual(mesh.dim, 1)
        self.assertEqual(mesh.num_cells[0], 20)
        self.assertEqual(mesh.box_lengths[0], 10.0)
        self.assertEqual(mesh.mesh_sizes[0], 0.5)
        self.assertEqual(mesh.boundary_condition, "periodic")
        
    def test_init_2d(self):
        """Test 2D mesh initialization."""
        mesh = Mesh([10.0, 5.0], [20, 10])
        self.assertEqual(mesh.dim, 2)
        np.testing.assert_array_equal(mesh.num_cells, np.array([20, 10]))
        np.testing.assert_array_equal(mesh.box_lengths, np.array([10.0, 5.0]))
        np.testing.assert_array_equal(mesh.mesh_sizes, np.array([0.5, 0.5]))
        
    def test_init_dimension_mismatch(self):
        """Test error when box_lengths and num_cells dimensions don't match."""
        with self.assertRaises(AssertionError):
            Mesh([10.0, 5.0], 20)
            
    def test_index_to_position_1d(self):
        """Test index to position conversion in 1D."""
        mesh = Mesh(10.0, 20)
        pos = mesh.index_to_position(10)
        self.assertAlmostEqual(pos[0], 4.75)  # (10-0.5)*0.5 = 4.75
        
    def test_index_to_position_2d(self):
        """Test index to position conversion in 2D."""
        mesh = Mesh([10.0, 5.0], [20, 10])
        pos = mesh.index_to_position([10, 5])
        np.testing.assert_array_almost_equal(pos, np.array([4.75, 2.25]))
        
    def test_index_to_position_wrong_dim(self):
        """Test error when wrong number of indices are provided."""
        mesh = Mesh([10.0, 5.0], [20, 10])
        with self.assertRaises(ValueError):
            mesh.index_to_position(10)
            
    def test_position_to_index_1d(self):
        """Test position to index conversion in 1D."""
        mesh = Mesh(10.0, 20)
        idx = mesh.position_to_index(4.75)
        self.assertEqual(idx[0], 11)  # floor(4.75/0.5 + 0.5) + 1 = 11
        
    def test_position_to_index_2d(self):
        """Test position to index conversion in 2D."""
        mesh = Mesh([10.0, 5.0], [20, 10])
        idx = mesh.position_to_index([4.75, 2.25])
        np.testing.assert_array_equal(idx, np.array([11, 6]))
        
    def test_position_to_index_periodic(self):
        """Test position to index with periodic boundary conditions."""
        mesh = Mesh(10.0, 20)
        idx = mesh.position_to_index(14.75)  # Outside the box
        self.assertEqual(idx[0], 11)  # Should wrap around to 4.75
        
    def test_position_to_index_wrong_dim(self):
        """Test error when wrong number of position coordinates are provided."""
        mesh = Mesh([10.0, 5.0], [20, 10])
        with self.assertRaises(ValueError):
            mesh.position_to_index(4.75)
            
    def test_laplacian_1d_periodic(self):
        """Test Laplacian matrix for 1D mesh with periodic boundary conditions."""
        mesh = Mesh(10.0, 4)
        lap = mesh.laplacian()
        
        # divided by 1/2.5**2
        expected = jnp.array([
            [-2.0, 1.0, 0.0, 1.0],
            [1.0, -2.0, 1.0, 0.0],
            [0.0, 1.0, -2.0, 1.0],
            [1.0, 0.0, 1.0, -2.0]
        ]) * (1/2.5**2)
        
        np.testing.assert_array_almost_equal(lap, expected)
        
    def test_laplacian_higher_dim(self):
        """Test error when trying to compute Laplacian for dimensions > 1."""
        mesh = Mesh([10.0, 5.0], [20, 10])
        with self.assertRaises(NotImplementedError):
            mesh.laplacian()

if __name__ == "__main__":
    unittest.main()
