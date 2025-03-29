import abc
from typing import Sequence, Optional, Callable, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


class ScoreModel(abc.ABC):
    """Base abstract class for score model implementations.
    
    A score model takes position (x) and velocity (v) inputs and outputs
    a score vector of the same length as v. The model supports batched inputs,
    where the first dimension is the batch dimension.
    """
    
    @abc.abstractmethod
    def __call__(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Compute the score for given position and velocity inputs.
        
        Args:
            x: Position array of shape (batch_size, dx) or (..., dx)
            v: Velocity array of shape (batch_size, dv) or (..., dv)
            
        Returns:
            Score vector of shape (batch_size, dv) or (..., dv), 
            matching the batch dimensions of the inputs
        """
        pass


class MLPScoreModel(nn.Module):
    """MLP-based implementation of a score model."""
    
    hidden_dims: Sequence[int]
    activation: Callable = nn.relu
    output_dim: Optional[int] = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Compute the score using an MLP network.
        
        Args:
            x: Position array of shape (batch_size, dx) or (..., dx)
            v: Velocity array of shape (batch_size, dv) or (..., dv)
            
        Returns:
            Score vector of shape (batch_size, dv) or (..., dv),
            matching the batch dimensions of the inputs
        """
        # Get output dimension (default to velocity dimension)
        output_dim = self.output_dim or v.shape[-1]
        
        # Concatenate x and v along the last dimension
        # This preserves any batch dimensions
        inputs = jnp.concatenate([x, v], axis=-1)
        
        # Pass through MLP layers
        for dim in self.hidden_dims:
            inputs = nn.Dense(features=dim)(inputs)
            inputs = self.activation(inputs)
        
        # Final layer to produce the score
        outputs = nn.Dense(features=output_dim)(inputs)
        
        return outputs


class ResNetBlock(nn.Module):
    """A ResNet block for the score model."""
    
    dim: int
    activation: Callable = nn.relu
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = self.activation(nn.Dense(features=self.dim)(x))
        if x.shape[-1] == self.dim:
            return x + y
        return x + nn.Dense(features=x.shape[-1])(y)

class ResNetScoreModel(nn.Module):
    """ResNet-based implementation of a score model."""
    
    hidden_dims: Sequence[int]  # Changed to match MLPScoreModel
    activation: Callable = nn.relu
    output_dim: Optional[int] = None
    num_blocks: int = 4  # Default number of ResNet blocks per hidden dimension
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Compute the score using a ResNet architecture.
        
        Args:
            x: Position array of shape (batch_size, dx) or (..., dx)
            v: Velocity array of shape (batch_size, dv) or (..., dv)
            
        Returns:
            Score vector of shape (batch_size, dv) or (..., dv),
            matching the batch dimensions of the inputs
        """
        # Get output dimension (default to velocity dimension)
        output_dim = self.output_dim or v.shape[-1]
        
        # Concatenate x and v along the last dimension
        # This preserves any batch dimensions
        inputs = jnp.concatenate([x, v], axis=-1)
        
        # Initial projection to first hidden dimension
        h = nn.Dense(features=self.hidden_dims[0])(inputs)
        h = self.activation(h)
        
        # Apply ResNet blocks for each hidden dimension
        for i, dim in enumerate(self.hidden_dims):
            for _ in range(self.num_blocks):
                h = ResNetBlock(dim=dim, activation=self.activation)(h)
        
        # Final layer to produce the score
        outputs = nn.Dense(features=output_dim)(h)
        
        return outputs


# Helper functions to instantiate models
def create_mlp_score_model(hidden_dims=(128, 128), activation=nn.relu, output_dim=None):
    """Create an MLP-based score model.
    
    Args:
        hidden_dims: Sequence of hidden dimensions
        activation: Activation function
        output_dim: Output dimension (if None, will use velocity dimension)
        
    Returns:
        An initialized MLPScoreModel instance
    """
    return MLPScoreModel(hidden_dims=hidden_dims, activation=activation, output_dim=output_dim)


def create_resnet_score_model(hidden_dims=(128, 128), activation=nn.relu, output_dim=None, num_blocks=4):
    """Create a ResNet-based score model.
    
    Args:
        hidden_dims: Sequence of hidden dimensions 
        activation: Activation function
        output_dim: Output dimension (if None, will use velocity dimension)
        num_blocks: Number of ResNet blocks per hidden dimension
        
    Returns:
        An initialized ResNetScoreModel instance
    """
    return ResNetScoreModel(
        hidden_dims=hidden_dims,
        activation=activation,
        output_dim=output_dim,
        num_blocks=num_blocks
    )
