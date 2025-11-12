import jax
import jax.numpy as jnp
from functools import partial
from flax import nnx
import orbax.checkpoint as ocp
import os
import jax.lax as lax

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

"KDE score model for 1D space"
def _silverman_bandwidth(v, eps=1e-12):
        n, dv = v.shape
        sigma = jnp.std(v, axis=0, ddof=1) + eps
        return sigma * n ** (-1.0 / (dv + 4.0))  # (dv,)

@partial(jax.jit, static_argnames=['ichunk', 'jchunk'])
def score_kde_blocked(x, v, cells, eta, eps=1e-12, hv=None, ichunk=2048, jchunk=2048):
    if hv is None: hv = _silverman_bandwidth(v, eps)
    L = eta * cells.size
    n, dv = v.shape
    inv_hv = 1.0 / hv
    inv_hv2 = inv_hv**2

    ni = (n + ichunk - 1) // ichunk
    nj = (n + jchunk - 1) // jchunk
    n_pad = ni * ichunk
    pad = n_pad - n

    x_pad = jnp.pad(x, (0, pad))
    v_pad = jnp.pad(v, ((0, pad), (0, 0)))
    u_pad = v_pad * inv_hv

    Zp = jnp.zeros((n_pad, 1), v.dtype)
    Mp = jnp.zeros((n_pad, dv), v.dtype)

    ar_i = jnp.arange(ichunk)
    ar_j = jnp.arange(jchunk)

    def loop_j(tj, carry2):
        Zi_, Mi_, Ri, Ui, Vi, Ui2 = carry2
        j0 = tj * jchunk
        mj = jnp.minimum(jchunk, n - j0)

        Rj = lax.dynamic_slice(x_pad, (j0,), (jchunk,))
        Uj = lax.dynamic_slice(u_pad, (j0, 0), (jchunk, dv))
        Vj = lax.dynamic_slice(v_pad, (j0, 0), (jchunk, dv))
        Uj2 = jnp.sum(Uj * Uj, axis=1, keepdims=True).T
        mask_j = (ar_j < mj).astype(v.dtype).reshape(1, jchunk)

        dx = Ri[:, None] - Rj[None, :]
        dx = (dx + 0.5 * L) % L - 0.5 * L
        psi = jnp.clip(1.0 - jnp.abs(dx) / eta, 0.0, 1.0)

        G = Ui @ Uj.T
        Kj = jnp.exp(G - 0.5 * Ui2 - 0.5 * Uj2)

        w = (psi * Kj + eps) * mask_j
        Zi_ = Zi_ + jnp.sum(w, axis=1, keepdims=True)
        Mi_ = Mi_ + w @ Vj
        return Zi_, Mi_, Ri, Ui, Vi, Ui2

    def loop_i(ti, carry):
        Zc, Mc = carry
        i0 = ti * ichunk
        mi = jnp.minimum(ichunk, n - i0)

        Ri = lax.dynamic_slice(x_pad, (i0,), (ichunk,))
        Ui = lax.dynamic_slice(u_pad, (i0, 0), (ichunk, dv))
        Vi = lax.dynamic_slice(v_pad, (i0, 0), (ichunk, dv))
        Ui2 = jnp.sum(Ui * Ui, axis=1, keepdims=True)

        Zi = jnp.zeros((ichunk, 1), v.dtype)
        Mi = jnp.zeros((ichunk, dv), v.dtype)

        Zi, Mi, *_ = lax.fori_loop(0, nj, loop_j, (Zi, Mi, Ri, Ui, Vi, Ui2))

        mask_i = (ar_i < mi).astype(v.dtype).reshape(ichunk, 1)
        Zi = Zi * mask_i
        Mi = Mi * mask_i

        Zc = lax.dynamic_update_slice(Zc, Zi, (i0, 0))
        Mc = lax.dynamic_update_slice(Mc, Mi, (i0, 0))
        return Zc, Mc

    Zp, Mp = lax.fori_loop(0, ni, loop_i, (Zp, Mp))
    Z = Zp[:n]
    M = Mp[:n]
    mu = M / Z
    return (mu - v) * inv_hv2