#%%
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
import optax
from functools import partial
import time
from tqdm import tqdm
import os
import json
import pickle
from datetime import datetime

# Import your modules
from src.score_model import MLPScoreModel, ResNetScoreModel
from src.density import CosineNormal
from src.loss import explicit_score_matching_loss, implicit_score_matching_loss

# Configuration variables for easy modification
LEARNING_RATE = 5e-4  # Learning rate for model training
NUM_EPOCHS = 100      # Number of training epochs
BATCH_SIZE = 64       # Batch size for training
HIDDEN_DIMS = [64, 64]  # Hidden dimensions for models - changed to list format for nnx
ACTIVATION = nnx.soft_sign     # Activation function - changed to nnx.soft_sign
NUM_PARTICLES = 2048   # Number of particles for sampling
DIV_MODE = 'reverse'  # Divergence mode for implicit score matching
ALPHA = 0.4           # Alpha parameter for CosineNormal
K = 0.5               # K parameter for CosineNormal
RANDOM_SEED = 42      # Random seed for reproducibility

# Test configurations
TEST_CONFIGS = [
    {
        "name": f"1d1d_{div_mode}",
        "dx": 1,
        "dv": 1,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_dims": HIDDEN_DIMS,
        "activation": ACTIVATION,
        "num_particles": NUM_PARTICLES,
        "div_mode": div_mode,
        "alpha": ALPHA,
        "k": K,
        "random_seed": RANDOM_SEED
    } for div_mode in ['forward']
]

# Create directories if they don't exist
def ensure_dir_exists(dir_path):
    """Ensure the directory exists."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def ensure_plots_dir_exists():
    """Ensure the plots directory exists."""
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
    return ensure_dir_exists(plots_dir)

def ensure_data_dir_exists():
    """Ensure the data directory exists."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    return ensure_dir_exists(data_dir)

def save_config(config, run_id):
    """Save configuration to JSON file."""
    data_dir = ensure_data_dir_exists()
    config_dir = os.path.join(data_dir, 'configs')
    ensure_dir_exists(config_dir)
    
    # Create a copy of the config that can be serialized to JSON
    json_safe_config = {}
    for key, value in config.items():
        if key == 'activation':
            # Store activation function name as string
            json_safe_config[key] = value.__name__
        elif isinstance(value, (int, float, str, bool, list, dict, tuple)) or value is None:
            # These types are JSON serializable
            json_safe_config[key] = value
        else:
            # Convert other types to string representation
            json_safe_config[key] = str(value)
    
    # Save config to JSON file
    config_path = os.path.join(config_dir, f"{config['name']}_{run_id}.json")
    with open(config_path, 'w') as f:
        json.dump(json_safe_config, f, indent=2)
    
    return config_path

def save_run_data(results, config, run_id):
    """Save run data to pickle file."""
    data_dir = ensure_data_dir_exists()
    results_dir = os.path.join(data_dir, 'results')
    ensure_dir_exists(results_dir)
    
    # Save results to pickle file
    results_path = os.path.join(results_dir, f"{config['name']}_{run_id}.pkl")
    
    # Create a modified copy of the config without unpicklable objects
    safe_config = {}
    for key, value in config.items():
        if key == 'activation':
            # Skip the activation function
            continue
        else:
            safe_config[key] = value
    
    # Convert JAX arrays to numpy for easier loading later
    numpy_results = {}
    for key, value in results.items():
        if key == 'data':
            X, V, true_scores = value
            numpy_results['data'] = {
                'X': np.array(X),
                'V': np.array(V),
                'true_scores': np.array(true_scores)
            }
        elif key == 'before_training' or key == 'explicit_sm' or key == 'implicit_sm':
            mlp_output, resnet_output = value
            numpy_results[key] = {
                'mlp': np.array(mlp_output),
                'resnet': np.array(resnet_output)
            }
        elif key == 'losses':
            numpy_results['losses'] = {
                k: np.array(v) for k, v in value.items()
            }
        elif key == 'metrics':
            # Extract only the numeric metrics, avoiding any function objects
            numpy_results['metrics'] = {}
            for model_name, metric_dict in value.items():
                numpy_results['metrics'][model_name] = {
                    'mse': float(metric_dict['mse']),
                    'component_mse': np.array(metric_dict['component_mse']),
                    # Omit 'predictions' to avoid possible function references
                }
        elif key == 'batch_times':
            numpy_results['batch_times'] = value
        else:
            numpy_results[key] = value
    
    # Add safe config info to the results
    numpy_results['config'] = safe_config
    
    with open(results_path, 'wb') as f:
        pickle.dump(numpy_results, f)
    
    return results_path

def create_train_state(model, learning_rate, key, x_sample, v_sample):
    """Create initial training state with model parameters and optimizer."""
    # Create optimizer - changed to AdamW
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate))
    
    return {
        'model': model,
        'optimizer': optimizer,
        'step': 0
    }

def count_parameters(module: nnx.Module) -> int:
    params = nnx.state(module, nnx.Param)
    return sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))

def print_model_info(model, model_name):
    """Print model architecture information."""
    total_params = count_parameters(model)
    print(f"\n{model_name} Architecture:")
    
    if hasattr(model, 'hidden_dims'):
        print(f"  Hidden dimensions: {model.hidden_dims}")
    elif hasattr(model, 'mlp') and hasattr(model.mlp, 'hidden_dims'):
        print(f"  MLP Hidden dimensions: {model.mlp.hidden_dims}")
    
    print(f"  Activation function: {model.activation.__name__ if hasattr(model, 'activation') else 'N/A'}")
    print(f"  Total parameters: {total_params:,}")

def train_explicit_score_matching(train_state, x_data, v_data, true_scores, 
                                  num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, key=None, model_name="Model"):
    """Train model using explicit score matching loss."""
    model = train_state['model']
    optimizer = train_state['optimizer']
    
    @nnx.jit
    def train_step(optimizer, x_batch, v_batch, score_batch):
        def loss_fn(model):
            def model_fn(x, v):
                return model(x, v)
            
            loss = explicit_score_matching_loss(model_fn, x_batch, v_batch, score_batch)
            return loss, model_fn(x_batch, v_batch)
        
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, _), grads = grad_fn(optimizer.model)
        optimizer.update(grads)
        return loss
    
    n_samples = x_data.shape[0]
    steps_per_epoch = n_samples // batch_size
    
    if key is None:
        key = jr.PRNGKey(0)
    
    losses = []
    batch_times = []  # Track time per batch
    
    print(f"Training {model_name} with Explicit Score Matching for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        key, subkey = jr.split(key)
        perm = jr.permutation(subkey, n_samples)
        x_shuffled = x_data[perm]
        v_shuffled = v_data[perm]
        scores_shuffled = true_scores[perm]
        
        epoch_losses = []
        epoch_batch_times = []
        
        for i in range(steps_per_epoch):
            batch_start_time = time.time()
            
            batch_idx = slice(i * batch_size, (i + 1) * batch_size)
            x_batch = x_shuffled[batch_idx]
            v_batch = v_shuffled[batch_idx]
            score_batch = scores_shuffled[batch_idx]
            
            loss = train_step(optimizer, x_batch, v_batch, score_batch)
            
            batch_time = time.time() - batch_start_time
            epoch_batch_times.append(batch_time)
            epoch_losses.append(loss)
        
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        avg_batch_time = np.mean(epoch_batch_times)
        batch_times.append(avg_batch_time)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{model_name} - Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Avg batch time: {avg_batch_time*1000:.2f} ms")
    
    training_time = time.time() - start_time
    avg_time_per_batch = np.mean(batch_times)
    print(f"{model_name} training completed in {training_time:.2f} seconds, avg time per batch: {avg_time_per_batch*1000:.2f} ms")
    
    train_state['step'] += num_epochs
    return train_state, losses, avg_time_per_batch

def train_implicit_score_matching(train_state, x_data, v_data, true_scores=None,
                                 num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, key=None, div_mode=DIV_MODE, model_name="Model"):
    """Train model using implicit score matching loss, also tracking explicit loss."""
    model = train_state['model']
    optimizer = train_state['optimizer']
    
    @nnx.jit
    def train_step(optimizer, x_batch, v_batch, subkey):
        def loss_fn(model):
            def model_fn(x, v):
                return model(x, v)
            
            loss = implicit_score_matching_loss(
                model_fn, x_batch, v_batch, 
                key=subkey, div_mode=div_mode, n_samples=10
            )
            return loss
        
        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(optimizer.model)
        optimizer.update(grads)
        return loss
    
    @nnx.jit
    def compute_explicit_loss(model, x_batch, v_batch, true_scores_batch):
        def model_fn(x, v):
            return model(x, v)
        
        return explicit_score_matching_loss(model_fn, x_batch, v_batch, true_scores_batch)
    
    n_samples = x_data.shape[0]
    steps_per_epoch = n_samples // batch_size
    
    if key is None:
        key = jr.PRNGKey(0)
    
    implicit_losses = []
    explicit_losses = [] # Track explicit losses too
    batch_times = []  # Track time per batch
    
    print(f"Training {model_name} with Implicit Score Matching for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        key, subkey = jr.split(key)
        perm = jr.permutation(subkey, n_samples)
        x_shuffled = x_data[perm]
        v_shuffled = v_data[perm]
        true_scores_shuffled = true_scores[perm] if true_scores is not None else None
        
        epoch_implicit_losses = []
        epoch_explicit_losses = []
        epoch_batch_times = []
        
        for i in range(steps_per_epoch):
            batch_start_time = time.time()
            
            batch_idx = slice(i * batch_size, (i + 1) * batch_size)
            x_batch = x_shuffled[batch_idx]
            v_batch = v_shuffled[batch_idx]
            key, subkey = jr.split(key)
            
            implicit_loss = train_step(optimizer, x_batch, v_batch, subkey)
            
            # Also compute explicit loss if true scores are available
            if true_scores is not None:
                true_scores_batch = true_scores_shuffled[batch_idx]
                explicit_loss = compute_explicit_loss(model, x_batch, v_batch, true_scores_batch)
                epoch_explicit_losses.append(explicit_loss)
            
            batch_time = time.time() - batch_start_time
            epoch_batch_times.append(batch_time)
            epoch_implicit_losses.append(implicit_loss)
        
        avg_implicit_loss = jnp.mean(jnp.array(epoch_implicit_losses))
        implicit_losses.append(avg_implicit_loss)
        
        if true_scores is not None:
            avg_explicit_loss = jnp.mean(jnp.array(epoch_explicit_losses))
            explicit_losses.append(avg_explicit_loss)
        
        avg_batch_time = np.mean(epoch_batch_times)
        batch_times.append(avg_batch_time)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if true_scores is not None:
                print(f"{model_name} - Epoch {epoch+1}/{num_epochs}, Implicit Loss: {avg_implicit_loss:.6f}, " +
                      f"Explicit Loss: {avg_explicit_loss:.6f}, Avg batch time: {avg_batch_time*1000:.2f} ms")
            else:
                print(f"{model_name} - Epoch {epoch+1}/{num_epochs}, Implicit Loss: {avg_implicit_loss:.6f}, " +
                      f"Avg batch time: {avg_batch_time*1000:.2f} ms")
    
    training_time = time.time() - start_time
    avg_time_per_batch = np.mean(batch_times)
    print(f"{model_name} training completed in {training_time:.2f} seconds, avg time per batch: {avg_time_per_batch*1000:.2f} ms")
    
    train_state['step'] += num_epochs
    return train_state, implicit_losses, explicit_losses if true_scores is not None else None, avg_time_per_batch

def evaluate_model(model, x_data, v_data, true_scores):
    """Evaluate model performance."""
    @nnx.jit
    def compute_predictions(model, x_data, v_data):
        def pred_fn(x, v):
            return model(x, v)
        
        return jax.vmap(pred_fn)(x_data, v_data)
    
    predicted_scores = compute_predictions(model, x_data, v_data)
    mse = jnp.mean(jnp.sum(jnp.square(predicted_scores - true_scores), axis=-1))
    component_mse = jnp.mean(jnp.square(predicted_scores - true_scores), axis=0)
    
    return {
        'mse': mse,
        'component_mse': component_mse,
        'predictions': predicted_scores
    }

def print_evaluation_results(results, model_name):
    """Print evaluation results."""
    print(f"\n--- {model_name} Evaluation Results ---")
    print(f"Overall MSE: {results['mse']:.6f}")
    print("Component-wise MSE:")
    for i, mse in enumerate(results['component_mse']):
        print(f"  Component {i+1}: {mse:.6f}")
    print()

def plot_training_curves(mlp_losses, resnet_losses, titles, config_name):
    """Plot training loss curves."""
    plots_dir = ensure_plots_dir_exists()
    fig, axes = plt.subplots(1, len(titles), figsize=(16, 6))
    
    for i, (mlp_loss, resnet_loss, title) in enumerate(zip(mlp_losses, resnet_losses, titles)):
        axes[i].plot(mlp_loss, label='MLP')
        axes[i].plot(resnet_loss, label='ResNet')
        axes[i].set_title(f'{title} Training Loss')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        
        if 'Explicit' in title:
            axes[i].set_yscale('log')
        else:
            axes[i].set_yscale('linear')
            
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'training_curves_{config_name}.png'))
    plt.show()

def plot_score_comparisons(X, V, true_scores, 
                           mlp_before, resnet_before, 
                           mlp_explicit, resnet_explicit,
                           mlp_implicit, resnet_implicit,
                           config,
                           run_id=""):
    """Plot comparisons of score models before and after training."""
    plots_dir = ensure_plots_dir_exists()
    config_name = config["name"]
    dv = config["dv"]
    
    # Add run_id to the filename if provided
    file_suffix = f"{config_name}_{run_id}" if run_id else config_name
    
    X_np = np.array(X)
    V_np = np.array(V)
    true_scores_np = np.array(true_scores)
    mlp_before_np = np.array(mlp_before)
    resnet_before_np = np.array(resnet_before)
    mlp_explicit_np = np.array(mlp_explicit)
    resnet_explicit_np = np.array(resnet_explicit)
    mlp_implicit_np = np.array(mlp_implicit)
    resnet_implicit_np = np.array(resnet_implicit)
    
    if dv == 1:
        plot_1d1d_scores(X_np, V_np, true_scores_np, 
                         mlp_before_np, resnet_before_np,
                         mlp_explicit_np, resnet_explicit_np, 
                         mlp_implicit_np, resnet_implicit_np,
                         file_suffix, plots_dir)
    else:
        for comp_idx in range(dv):
            plot_scores_by_x(X_np, true_scores_np, 
                            mlp_before_np, resnet_before_np,
                            mlp_explicit_np, resnet_explicit_np, 
                            mlp_implicit_np, resnet_implicit_np,
                            comp_idx, file_suffix, plots_dir)

def plot_1d1d_scores(X, V, true_scores, 
                     mlp_before, resnet_before,
                     mlp_explicit, resnet_explicit, 
                     mlp_implicit, resnet_implicit,
                     file_suffix, plots_dir):
    """Plot scores with v on horizontal axis for the 1d1d case."""
    fig, axes = plt.subplots(4, 2, figsize=(18, 24))
    sort_idx = np.argsort(V.flatten())
    V_sorted = V.flatten()[sort_idx]
    
    for i, (title, mlp_data, resnet_data) in enumerate([
        ("Before Training", mlp_before, resnet_before),
        ("After Explicit SM", mlp_explicit, resnet_explicit),
        ("After Implicit SM", mlp_implicit, resnet_implicit)
    ]):
        ax = axes[i, 0]
        true_score = true_scores.flatten()[sort_idx]
        pred_score = mlp_data.flatten()[sort_idx]
        ax.plot(V_sorted, true_score, 'b-', linewidth=2, label='True Score')
        ax.plot(V_sorted, pred_score, 'r--', linewidth=2, label='MLP Prediction')
        ax.set_title(f'{title} - MLP')
        ax.set_xlabel('Velocity (v)')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[i, 1]
        pred_score = resnet_data.flatten()[sort_idx]
        ax.plot(V_sorted, true_score, 'b-', linewidth=2, label='True Score')
        ax.plot(V_sorted, pred_score, 'r--', linewidth=2, label='ResNet Prediction')
        ax.set_title(f'{title} - ResNet')
        ax.set_xlabel('Velocity (v)')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    ax = axes[3, 0]
    ax.plot(V_sorted, np.abs(true_scores.flatten()[sort_idx] - mlp_explicit.flatten()[sort_idx]), 'g-', label='Explicit SM')
    ax.plot(V_sorted, np.abs(true_scores.flatten()[sort_idx] - mlp_implicit.flatten()[sort_idx]), 'r-', label='Implicit SM')
    ax.set_title('MLP Absolute Error')
    ax.set_xlabel('Velocity (v)')
    ax.set_ylabel('Absolute Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[3, 1]
    ax.plot(V_sorted, np.abs(true_scores.flatten()[sort_idx] - resnet_explicit.flatten()[sort_idx]), 'g-', label='Explicit SM')
    ax.plot(V_sorted, np.abs(true_scores.flatten()[sort_idx] - resnet_implicit.flatten()[sort_idx]), 'r-', label='Implicit SM')
    ax.set_title('ResNet Absolute Error')
    ax.set_xlabel('Velocity (v)')
    ax.set_ylabel('Absolute Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'score_comparison_{file_suffix}.png'))
    plt.show()

def plot_scores_by_x(X, true_scores, 
                     mlp_before, resnet_before,
                     mlp_explicit, resnet_explicit, 
                     mlp_implicit, resnet_implicit,
                     component_idx, file_suffix, plots_dir):
    """Plot scores with x on horizontal axis for higher dimensional cases."""
    fig, axes = plt.subplots(4, 2, figsize=(18, 24))
    sort_idx = np.argsort(X.flatten())
    X_sorted = X.flatten()[sort_idx]
    
    for i, (title, mlp_data, resnet_data) in enumerate([
        ("Before Training", mlp_before, resnet_before),
        ("After Explicit SM", mlp_explicit, resnet_explicit),
        ("After Implicit SM", mlp_implicit, resnet_implicit)
    ]):
        ax = axes[i, 0]
        true_comp = true_scores[:, component_idx][sort_idx]
        pred_comp = mlp_data[:, component_idx][sort_idx]
        ax.plot(X_sorted, true_comp, 'b-', linewidth=2, label='True Score')
        ax.plot(X_sorted, pred_comp, 'r--', linewidth=2, label='MLP Prediction')
        ax.set_title(f'{title} - MLP - Component {component_idx+1}')
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[i, 1]
        pred_comp = resnet_data[:, component_idx][sort_idx]
        ax.plot(X_sorted, true_comp, 'b-', linewidth=2, label='True Score')
        ax.plot(X_sorted, pred_comp, 'r--', linewidth=2, label='ResNet Prediction')
        ax.set_title(f'{title} - ResNet - Component {component_idx+1}')
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    ax = axes[3, 0]
    ax.plot(X_sorted, np.abs(true_scores[:, component_idx][sort_idx] - mlp_explicit[:, component_idx][sort_idx]), 'g-', label='Explicit SM')
    ax.plot(X_sorted, np.abs(true_scores[:, component_idx][sort_idx] - mlp_implicit[:, component_idx][sort_idx]), 'r-', label='Implicit SM')
    ax.set_title(f'MLP Absolute Error - Component {component_idx+1}')
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Absolute Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[3, 1]
    ax.plot(X_sorted, np.abs(true_scores[:, component_idx][sort_idx] - resnet_explicit[:, component_idx][sort_idx]), 'g-', label='Explicit SM')
    ax.plot(X_sorted, np.abs(true_scores[:, component_idx][sort_idx] - resnet_implicit[:, component_idx][sort_idx]), 'r-', label='Implicit SM')
    ax.set_title(f'ResNet Absolute Error - Component {component_idx+1}')
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Absolute Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'score_component{component_idx+1}_{file_suffix}.png'))
    plt.show()

def run_test(config):
    """Run a full test for a single configuration."""
    # Generate a unique run ID with timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the config first
    config_path = save_config(config, run_id)
    print(f"Config saved to: {config_path}")
    
    dx = config["dx"]
    dv = config["dv"]
    config_name = config["name"]
    
    print(f"\n=== Running test for dx={dx}, dv={dv} configuration ===")
    
    # Use config parameters instead of global constants
    density = CosineNormal(alpha=config["alpha"], k=config["k"], dx=dx, dv=dv)
    key1, key2 = jr.split(jr.PRNGKey(config["random_seed"]))
    X, V = density.sample(key1, size=config["num_particles"])
    true_scores = density.score(X, V)
    
    print(f"Input shapes - X: {X.shape}, V: {V.shape}")
    print(f"True scores shape: {true_scores.shape}")
    
    # Create models using the nnx API directly
    key_mlp, key_mlp_in_resnet, key_resnet, key_mlp_implicit, key_mlp_in_resnet_implicit, key_resnet_implicit = jr.split(key2, 6)
    
    # Create MLP models - separate instances for standalone and for ResNet
    mlp_model = MLPScoreModel(
        dx=dx,
        dv=dv,
        hidden_dims=config["hidden_dims"],
        activation=config["activation"],
        seed=int(jr.randint(key_mlp, (), 0, 1000000))
    )
    
    # Create a separate MLP for the ResNet model
    mlp_in_resnet = MLPScoreModel(
        dx=dx,
        dv=dv,
        hidden_dims=config["hidden_dims"],
        activation=config["activation"],
        seed=int(jr.randint(key_mlp_in_resnet, (), 0, 1000000))
    )
    
    # Create ResNet model with the separate MLP
    resnet_model = ResNetScoreModel(
        mlp=mlp_in_resnet,  # Use the dedicated MLP
        seed=int(jr.randint(key_resnet, (), 0, 1000000))
    )
    
    # Print model architecture information
    print_model_info(mlp_model, "MLP")
    print_model_info(resnet_model, "ResNet")
    
    # Create train states
    mlp_state_explicit = create_train_state(mlp_model, config["learning_rate"], key_mlp, X[0:1], V[0:1])
    resnet_state_explicit = create_train_state(resnet_model, config["learning_rate"], key_resnet, X[0:1], V[0:1])
    
    # Create new instances for implicit training
    mlp_model_implicit = MLPScoreModel(
        dx=dx,
        dv=dv,
        hidden_dims=config["hidden_dims"],
        activation=config["activation"],
        seed=int(jr.randint(key_mlp_implicit, (), 0, 1000000))
    )
    
    # Create a separate MLP for the implicit ResNet model
    mlp_in_resnet_implicit = MLPScoreModel(
        dx=dx,
        dv=dv,
        hidden_dims=config["hidden_dims"],
        activation=config["activation"],
        seed=int(jr.randint(key_mlp_in_resnet_implicit, (), 0, 1000000))
    )
    
    resnet_model_implicit = ResNetScoreModel(
        mlp=mlp_in_resnet_implicit,  # Use the dedicated MLP for implicit training
        seed=int(jr.randint(key_resnet_implicit, (), 0, 1000000))
    )
    
    mlp_state_implicit = create_train_state(mlp_model_implicit, config["learning_rate"], key_mlp_implicit, X[0:1], V[0:1])
    resnet_state_implicit = create_train_state(resnet_model_implicit, config["learning_rate"], key_resnet_implicit, X[0:1], V[0:1])
    
    # Compute initial outputs (before training)
    @nnx.jit
    def compute_output(model, x, v):
        def pred_fn(x, v):
            return model(x, v)
        return jax.vmap(pred_fn)(x, v)
    
    mlp_output_before = compute_output(mlp_model, X, V)
    resnet_output_before = compute_output(resnet_model, X, V)
    
    print("Evaluating models before training...")
    mlp_results_before = evaluate_model(mlp_model, X, V, true_scores)
    resnet_results_before = evaluate_model(resnet_model, X, V, true_scores)
    
    print_evaluation_results(mlp_results_before, "MLP Before Training")
    print_evaluation_results(resnet_results_before, "ResNet Before Training")
    
    print("\n=== Training with Explicit Score Matching ===")
    mlp_state_explicit, mlp_explicit_losses, mlp_explicit_batch_time = train_explicit_score_matching(
        mlp_state_explicit, X, V, true_scores, 
        num_epochs=config["num_epochs"], batch_size=config["batch_size"],
        model_name="MLP"
    )
    
    resnet_state_explicit, resnet_explicit_losses, resnet_explicit_batch_time = train_explicit_score_matching(
        resnet_state_explicit, X, V, true_scores,
        num_epochs=config["num_epochs"], batch_size=config["batch_size"],
        model_name="ResNet"
    )
    
    print("\n=== Training with Implicit Score Matching ===")
    mlp_state_implicit, mlp_implicit_losses, mlp_explicit_during_implicit, mlp_implicit_batch_time = train_implicit_score_matching(
        mlp_state_implicit, X, V, true_scores,
        num_epochs=config["num_epochs"], batch_size=config["batch_size"],
        div_mode=config["div_mode"],
        model_name="MLP"
    )
    
    resnet_state_implicit, resnet_implicit_losses, resnet_explicit_during_implicit, resnet_implicit_batch_time = train_implicit_score_matching(
        resnet_state_implicit, X, V, true_scores,
        num_epochs=config["num_epochs"], batch_size=config["batch_size"],
        div_mode=config["div_mode"],
        model_name="ResNet"
    )
    
    # Compute outputs after training
    mlp_output_explicit = compute_output(mlp_state_explicit['model'], X, V)
    resnet_output_explicit = compute_output(resnet_state_explicit['model'], X, V)
    
    mlp_output_implicit = compute_output(mlp_state_implicit['model'], X, V)
    resnet_output_implicit = compute_output(resnet_state_implicit['model'], X, V)
    
    print("\nEvaluating models after training...")
    
    mlp_results_explicit = evaluate_model(mlp_state_explicit['model'], X, V, true_scores)
    resnet_results_explicit = evaluate_model(resnet_state_explicit['model'], X, V, true_scores)
    
    mlp_results_implicit = evaluate_model(mlp_state_implicit['model'], X, V, true_scores)
    resnet_results_implicit = evaluate_model(resnet_state_implicit['model'], X, V, true_scores)
    
    print_evaluation_results(mlp_results_explicit, "MLP with Explicit Score Matching")
    print_evaluation_results(resnet_results_explicit, "ResNet with Explicit Score Matching")
    print_evaluation_results(mlp_results_implicit, "MLP with Implicit Score Matching")
    print_evaluation_results(resnet_results_implicit, "ResNet with Implicit Score Matching")
    
    # Add run_id to the plot filenames
    plot_training_curves(
        [mlp_explicit_losses, mlp_implicit_losses, mlp_explicit_during_implicit],
        [resnet_explicit_losses, resnet_implicit_losses, resnet_explicit_during_implicit],
        ["Explicit Score Matching", "Implicit Score Matching", "Explicit Loss During Implicit Training"],
        f"{config_name}_{run_id}"
    )
    
    plot_score_comparisons(
        X, V, true_scores,
        mlp_output_before, resnet_output_before,
        mlp_output_explicit, resnet_output_explicit,
        mlp_output_implicit, resnet_output_implicit,
        config,
        run_id
    )
    
    # Collect results
    results = {
        'data': (X, V, true_scores),
        'before_training': (mlp_output_before, resnet_output_before),
        'explicit_sm': (mlp_output_explicit, resnet_output_explicit),
        'implicit_sm': (mlp_output_implicit, resnet_output_implicit),
        'losses': {
            'mlp_explicit': mlp_explicit_losses,
            'resnet_explicit': resnet_explicit_losses,
            'mlp_implicit': mlp_implicit_losses,
            'resnet_implicit': resnet_implicit_losses,
            'mlp_explicit_during_implicit': mlp_explicit_during_implicit,
            'resnet_explicit_during_implicit': resnet_explicit_during_implicit
        },
        'metrics': {
            'mlp_explicit': {
                'mse': float(mlp_results_explicit['mse']),
                'component_mse': np.array(mlp_results_explicit['component_mse']),
            },
            'resnet_explicit': {
                'mse': float(resnet_results_explicit['mse']),
                'component_mse': np.array(resnet_results_explicit['component_mse']),
            },
            'mlp_implicit': {
                'mse': float(mlp_results_implicit['mse']),
                'component_mse': np.array(mlp_results_implicit['component_mse']),
            },
            'resnet_implicit': {
                'mse': float(resnet_results_implicit['mse']),
                'component_mse': np.array(resnet_results_implicit['component_mse']),
            }
        },
        'batch_times': {
            'mlp_explicit': mlp_explicit_batch_time,
            'resnet_explicit': resnet_explicit_batch_time,
            'mlp_implicit': mlp_implicit_batch_time,
            'resnet_implicit': resnet_implicit_batch_time
        }
    }
    
    # Print a summary of batch times
    print("\n=== Training Performance Summary ===")
    print(f"MLP Explicit Score Matching: {mlp_explicit_batch_time*1000:.2f} ms/batch")
    print(f"ResNet Explicit Score Matching: {resnet_explicit_batch_time*1000:.2f} ms/batch")
    print(f"MLP Implicit Score Matching: {mlp_implicit_batch_time*1000:.2f} ms/batch")
    print(f"ResNet Implicit Score Matching: {resnet_implicit_batch_time*1000:.2f} ms/batch")
    
    # Save the results
    results_path = save_run_data(results, config, run_id)
    print(f"Results saved to: {results_path}")
    
    return results, run_id

def main():
    results = {}
    run_ids = {}
    
    for config in TEST_CONFIGS:
        results[config["name"]], run_ids[config["name"]] = run_test(config)
    
    # Save a summary of all runs
    data_dir = ensure_data_dir_exists()
    summary = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "configurations": [
            {
                "name": config["name"],
                "run_id": run_ids[config["name"]],
                "dx": config["dx"],
                "dv": config["dv"]
            } for config in TEST_CONFIGS
        ]
    }
    
    summary_path = os.path.join(data_dir, f"run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nRun summary saved to: {summary_path}")
    
    return results
    
if __name__ == "__main__":
    results = main()

# %%
