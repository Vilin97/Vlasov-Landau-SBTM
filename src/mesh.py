from typing import Union, List, Tuple, Iterator
import jax.numpy as jnp
from itertools import product

class Mesh:
    def __init__(
        self, 
        box_lengths: Union[float, List[float], jnp.ndarray], 
        num_cells: Union[int, List[int], jnp.ndarray],
        boundary_condition: str = "periodic"
    ):
        # Convert inputs to arrays
        self.box_lengths = jnp.atleast_1d(jnp.asarray(box_lengths))
        self.num_cells = jnp.atleast_1d(jnp.asarray(num_cells, dtype=jnp.int32))
        self.dim = len(self.box_lengths)
        
        # Check dimensions match
        assert self.box_lengths.shape == self.num_cells.shape, "Box lengths and number of cells must have the same dimension"
        
        # Calculate mesh sizes
        self.mesh_sizes = self.box_lengths / self.num_cells
        
        # Set boundary condition
        self.boundary_condition = boundary_condition
        
        # Pre-compute Laplacian matrix
        self._laplacian_matrix = self._build_laplacian()
    
    def _build_laplacian(self) -> jnp.ndarray:
        """Helper method to build the discrete Laplacian matrix Λ."""
        if self.dim > 1:
            raise NotImplementedError("Laplacian for dimensions > 1 is not implemented")
        
        n = self.num_cells[0]
        eta_squared = self.mesh_sizes[0] ** 2
        
        # Create indices for the matrix
        i, j = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing='ij')
        
        # Set diagonal and off-diagonal elements
        Lambda = jnp.where(i == j, -2.0, 0.0)
        
        if self.boundary_condition == "periodic":
            Lambda = jnp.where(((i - j) % n == 1) | ((j - i) % n == 1), 1.0, Lambda)
        else:
            Lambda = jnp.where(jnp.abs(i - j) == 1, 1.0, Lambda)
        
        # Scale by 1/η²
        return Lambda / eta_squared
    
    def index_to_position(self, indices: Union[int, Tuple[int, ...], List[int], jnp.ndarray]) -> jnp.ndarray:
        """Convert cell indices to physical positions."""
        indices = jnp.atleast_1d(jnp.asarray(indices))
        
        if indices.shape[0] != self.dim:
            raise ValueError(f"Expected {self.dim} indices, got {indices.shape[0]}")
        
        return (indices - 0.5) * self.mesh_sizes
    
    def position_to_index(self, positions: Union[float, Tuple[float, ...], List[float], jnp.ndarray]) -> jnp.ndarray:
        """Convert physical positions to cell indices."""
        positions = jnp.atleast_1d(jnp.asarray(positions))
        
        if positions.shape[0] != self.dim:
            raise ValueError(f"Expected {self.dim} position coordinates, got {positions.shape[0]}")
        
        # Handle periodic boundary conditions
        if self.boundary_condition == "periodic":
            positions = positions % self.box_lengths
        
        return jnp.floor(positions / self.mesh_sizes + 0.5).astype(jnp.int32) + 1
    
    def laplacian(self) -> jnp.ndarray:
        """Return the pre-computed discrete Laplacian matrix Λ."""
        return self._laplacian_matrix

    def __len__(self) -> int:
        """Return the total number of cells in the mesh."""
        return int(jnp.prod(self.num_cells))
    
