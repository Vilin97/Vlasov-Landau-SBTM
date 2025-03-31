import unittest
import jax.numpy as jnp
import numpy as np
import sys
import os

from src.mesh import Mesh, Mesh1D

class TestMesh1D(unittest.TestCase):
    def test_init_1d(self):
        """Test 1D mesh initialization."""
        mesh = Mesh1D(10.0, 20)
        self.assertEqual(mesh.dim, 1)
        self.assertEqual(mesh.num_cells[0], 20)
        self.assertEqual(mesh.box_lengths[0], 10.0)
        self.assertEqual(mesh.eta[0], 0.5)
        self.assertEqual(mesh.boundary_condition, "periodic")
    
    def test_index_to_position_1d(self):
        """Test index to position conversion in 1D."""
        mesh = Mesh1D(10.0, 20)
        pos = mesh.index_to_position(10)
        self.assertAlmostEqual(pos[0], 5.25)  # (10+0.5)*0.5 = 4.75
    
    def test_laplacian_1d_periodic(self):
        """Test Laplacian matrix for 1D mesh with periodic boundary conditions."""
        mesh = Mesh1D(10.0, 4)
        lap = mesh.laplacian()
        
        # divided by 1/2.5**2
        expected = jnp.array([
            [-2.0, 1.0, 0.0, 1.0],
            [1.0, -2.0, 1.0, 0.0],
            [0.0, 1.0, -2.0, 1.0],
            [1.0, 0.0, 1.0, -2.0]
        ]) * (1/2.5**2)
        
        np.testing.assert_array_almost_equal(lap, expected)
    
    def test_cells(self):
        """Test getting array of all cell positions."""
        mesh = Mesh1D(10.0, 4)
        positions = mesh.cells()
        
        expected = jnp.array([[1.25], [3.75], [6.25], [8.75]])
        self.assertEqual(positions.shape, (4, 1))
        np.testing.assert_array_almost_equal(positions, expected)
    
    def test_iteration(self):
        """Test iteration over cell positions."""
        mesh = Mesh1D(10.0, 4)
        positions = [pos for pos in mesh]
        
        expected = [jnp.array([1.25]), jnp.array([3.75]), jnp.array([6.25]), jnp.array([8.75])]
        self.assertEqual(len(positions), 4)
        
        for p, e in zip(positions, expected):
            np.testing.assert_array_almost_equal(p, e)
    
    def test_len(self):
        """Test length (number of cells)."""
        mesh = Mesh1D(10.0, 4)
        self.assertEqual(len(mesh), 4)

# This would be a test class for the general Mesh class when it becomes useful
# Currently, Mesh is abstract, so we can't instantiate it directly
class TestGenericMesh(unittest.TestCase):
    pass

if __name__ == "__main__":
    unittest.main()
