import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
import os

class MLPScoreModel(nnx.Module):
    """MLP-based implementation of a score model."""
    
    def save(self, path, **kwargs):
        os.makedirs(path, exist_ok=True)
        _, state = nnx.split(self)
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(path + '/state', state, **kwargs)
        return None

    def load(self, path):
        path += '/state'
        graphdef, abstract_state = nnx.split(self)
        checkpointer = ocp.StandardCheckpointer()
        state_restored = checkpointer.restore(path, abstract_state)

        # Get a new model instance with the restored weights
        restored = nnx.merge(graphdef, state_restored)

        # In-place update of known fields
        self.layers = restored.layers
        self.final_layer = restored.final_layer
        return None

    
    def __init__(self, dx, dv, hidden_dims=[128, 128], activation=nnx.soft_sign, seed=0, dtype=jnp.float32):
        """Initialize MLP score model.
        
        Args:
            dx: Dimension of position (x) input
            dv: Dimension of velocity (v) input
            hidden_dims: Sequence of hidden dimensions
            activation: Activation function
            seed: Random seed for initialization
            dtype: Data type for all layers (e.g., jnp.float32 or jnp.float64)
        """
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dtype = dtype
        rngs = nnx.Rngs(seed)
        
        # Initialize layers immediately
        self.layers = []
        input_dim = dx + dv  # Concatenated input dimensions
        
        for hidden_dim in self.hidden_dims:
            self.layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs, dtype=self.dtype))
            input_dim = hidden_dim
        
        self.final_layer = nnx.Linear(input_dim, dv, rngs=rngs, dtype=self.dtype)
    
    def __call__(self, x, v):
        """Compute the score using an MLP network.
        
        Args:
            x: Position array of shape (batch_size, dx) or (..., dx)
            v: Velocity array of shape (batch_size, dv) or (..., dv)
            
        Returns:
            Score vector of shape (batch_size, dv) or (..., dv),
            matching the batch dimensions of the inputs
        """
        # Concatenate x and v along the last dimension
        inputs = jnp.concatenate([x, v], axis=-1)
        
        # Pass through MLP layers
        h = inputs
        for layer in self.layers:
            h = self.activation(layer(h))
        
        # Final layer to produce the score
        outputs = self.final_layer(h)
        
        return outputs


class ResNetScoreModel(nnx.Module):
    """ResNet-based implementation of a score model."""
    
    def __init__(self, mlp, seed=0, dtype=jnp.float32):
        """Initialize ResNet score model using an MLP with residual connections.
        
        Args:
            mlp: An MLPScoreModel instance to use as the base network
            seed: Random seed for initialization
            dtype: Data type for all layers (e.g., jnp.float32 or jnp.float64)
        """
        self.mlp = mlp
        self.activation = mlp.activation
        self.rngs = nnx.Rngs(seed)
        self.dtype = dtype
        
        # Create projections for residual connections
        self.projections = []
        
        # Get the dimensions from the first layer in the MLP
        if len(self.mlp.layers) > 0:
            input_dim = self.mlp.layers[0].kernel.shape[0]  # Input dimension of first layer
            
            for layer in self.mlp.layers:
                out_dim = layer.kernel.shape[1]  # Output dimension of the layer
                
                # If dimensions don't match, create a projection
                if input_dim != out_dim:
                    projection = nnx.Linear(input_dim, out_dim, rngs=self.rngs, dtype=self.dtype)
                    self.projections.append(projection)
                else:
                    self.projections.append(None)
                
                input_dim = out_dim
    
    def __call__(self, x, v):
        """Compute the score using a ResNet architecture.
        
        Args:
            x: Position array of shape (batch_size, dx) or (..., dx)
            v: Velocity array of shape (batch_size, dv) or (..., dv)
            
        Returns:
            Score vector of shape (batch_size, dv) or (..., dv),
            matching the batch dimensions of the inputs
        """
        # Concatenate x and v along the last dimension
        inputs = jnp.concatenate([x, v], axis=-1)
        
        # Apply ResNet layers with skip connections
        h = inputs
        
        # Apply each layer with its corresponding projection/skip connection
        for i, (layer, projection) in enumerate(zip(self.mlp.layers, self.projections)):
            layer_output = self.activation(layer(h))
            
            # Apply projection if dimensions don't match, otherwise add directly
            if projection is not None:
                h = layer_output + projection(h)
            else:
                h = layer_output + h
        
        # Final layer to produce the score
        outputs = self.mlp.final_layer(h)
        
        return outputs
