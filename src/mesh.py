from typing import Union, List, Tuple, Iterator, Optional, Protocol, runtime_checkable
from abc import ABC, abstractmethod
import jax.numpy as jnp
from itertools import product

class Mesh(ABC):
    """Abstract base class for mesh implementations."""
    
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
        self.eta = self.box_lengths / self.num_cells
        
        # Set boundary condition
        self.boundary_condition = boundary_condition
    
    def index_to_position(self, indices: Union[int, Tuple[int, ...], List[int], jnp.ndarray]) -> jnp.ndarray:
        """Convert cell indices to physical positions."""
        indices = jnp.atleast_1d(jnp.asarray(indices))
        
        if indices.shape[0] != self.dim:
            raise ValueError(f"Expected {self.dim} indices, got {indices.shape[0]}")
        
        return (indices + 0.5) * self.eta
    
    def laplacian(self) -> jnp.ndarray:
        """Build the discrete Laplacian matrix for the mesh."""
        pass

    def __len__(self) -> int:
        """Return the total number of cells in the mesh."""
        return int(jnp.prod(self.num_cells))
    
    def __iter__(self) -> Iterator[jnp.ndarray]:
        """Iterator over all cell positions in the mesh."""
        # Create ranges for each dimension
        ranges = [range(n) for n in self.num_cells]
        
        # Use itertools.product to generate combinations of indices
        for indices in product(*ranges):
            yield self.index_to_position(indices)
    
class Mesh1D(Mesh):
    """One-dimensional mesh implementation."""
    
    def laplacian(self) -> jnp.ndarray:
        """Discrete Laplacian matrix."""
        n = self.num_cells[0]
        eta_squared = self.eta[0] ** 2
        
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
        
    # TODO: precompute this
    def cells(self) -> jnp.ndarray:
        """Return a JAX array containing the positions of all cells.
        
        Returns:
            jnp.ndarray: Array of shape (num_cells, 1) containing the
                         physical positions of all cells in the 1D mesh.
        """
        indices = jnp.arange(self.num_cells[0]).reshape(-1, 1)
        positions = (indices + 0.5) * self.eta[0]
        return positions

