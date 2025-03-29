#%%
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import flax.linen as nn
import optax
from functools import partial
import time
from tqdm import tqdm
import os
import json
import pickle
from datetime import datetime

# Import your modules
from src.score_model import create_mlp_score_model, create_resnet_score_model
from src.density import CosineNormal
from src.loss import explicit_score_matching_loss, implicit_score_matching_loss

# Configuration variables for easy modification
LEARNING_RATE = 5e-4  # Learning rate for model training
NUM_EPOCHS = 200      # Number of training epochs
BATCH_SIZE = 64       # Batch size for training
HIDDEN_DIMS = (64, 128, 64)  # Hidden dimensions for models
ACTIVATION = nn.soft_sign     # Activation function
NUM_BLOCKS = 1        # Number of blocks for ResNet
NUM_PARTICLES = 512   # Number of particles for sampling
DIV_MODE = 'reverse'  # Divergence mode for implicit score matching
ALPHA = 0.4           # Alpha parameter for CosineNormal
K = 0.5               # K parameter for CosineNormal
RANDOM_SEED = 42      # Random seed for reproducibility

# Test configurations
TEST_CONFIGS = [
    {
        "name": "1d1d",
        "dx": 1,
        "dv": 1,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_dims": HIDDEN_DIMS,
        "activation": ACTIVATION,
        "num_blocks": NUM_BLOCKS,
        "num_particles": NUM_PARTICLES,
        "div_mode": DIV_MODE,
        "alpha": ALPHA,
        "k": K,
        "random_seed": RANDOM_SEED
    },
    {
        "name": "1d2d",
        "dx": 1,
        "dv": 2,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_dims": HIDDEN_DIMS,
        "activation": ACTIVATION,
        "num_blocks": NUM_BLOCKS,
        "num_particles": NUM_PARTICLES,
        "div_mode": DIV_MODE,
        "alpha": ALPHA,
        "k": K,
        "random_seed": RANDOM_SEED
    }
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
        else:
            numpy_results[key] = value
    
    # Add config info to the results
    numpy_results['config'] = config
    
    with open(results_path, 'wb') as f:
        pickle.dump(numpy_results, f)
    
    return results_path

def create_train_state(model, learning_rate, key, x_sample, v_sample):
    """Create initial training state with model parameters and optimizer."""
    # Initialize model parameters
    params = model.init(key, x_sample, v_sample)
    
    # Create optimizer - changed to AdamW
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(params)
    
    return {
        'params': params,
        'opt_state': opt_state,
        'optimizer': optimizer,
        'step': 0
    }

def train_explicit_score_matching(model, train_state, x_data, v_data, true_scores, 
                                  num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, key=None, model_name="Model"):
    """Train model using explicit score matching loss."""
    
    @jax.jit
    def compute_loss_and_grads(params, x_batch, v_batch, score_batch):
        def loss_fn(params):
            def model_fn(x, v):
                return model.apply(params, x, v)
            loss = explicit_score_matching_loss(model_fn, x_batch, v_batch, score_batch)
            return loss
        loss, grads = jax.value_and_grad(loss_fn)(params)
        return loss, grads
    
    def train_step(state, x_batch, v_batch, score_batch):
        loss, grads = compute_loss_and_grads(state['params'], x_batch, v_batch, score_batch)
        updates, opt_state = state['optimizer'].update(grads, state['opt_state'], state['params'])
        params = optax.apply_updates(state['params'], updates)
        new_state = {
            'params': params,
            'opt_state': opt_state,
            'optimizer': state['optimizer'],
            'step': state['step'] + 1
        }
        return new_state, loss
    
    n_samples = x_data.shape[0]
    steps_per_epoch = n_samples // batch_size
    
    if key is None:
        key = jrandom.PRNGKey(0)
    
    losses = []
    print(f"Training {model_name} with Explicit Score Matching for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        key, subkey = jrandom.split(key)
        perm = jrandom.permutation(subkey, n_samples)
        x_shuffled = x_data[perm]
        v_shuffled = v_data[perm]
        scores_shuffled = true_scores[perm]
        
        epoch_losses = []
        for i in range(steps_per_epoch):
            batch_idx = slice(i * batch_size, (i + 1) * batch_size)
            x_batch = x_shuffled[batch_idx]
            v_batch = v_shuffled[batch_idx]
            score_batch = scores_shuffled[batch_idx]
            train_state, loss = train_step(train_state, x_batch, v_batch, score_batch)
            epoch_losses.append(loss)
        
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{model_name} - Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    training_time = time.time() - start_time
    print(f"{model_name} training completed in {training_time:.2f} seconds")
    
    return train_state, losses

def train_implicit_score_matching(model, train_state, x_data, v_data, 
                                 num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, key=None, div_mode=DIV_MODE, model_name="Model"):
    """Train model using implicit score matching loss."""
    
    @jax.jit
    def compute_loss_and_grads(params, x_batch, v_batch, subkey):
        def loss_fn(params):
            def model_fn(x, v):
                return model.apply(params, x, v)
            loss = implicit_score_matching_loss(
                model_fn, x_batch, v_batch, 
                key=subkey, div_mode=div_mode, n_samples=10
            )
            return loss
        loss, grads = jax.value_and_grad(loss_fn)(params)
        return loss, grads
    
    def train_step(state, x_batch, v_batch, subkey):
        loss, grads = compute_loss_and_grads(state['params'], x_batch, v_batch, subkey)
        updates, opt_state = state['optimizer'].update(grads, state['opt_state'], state['params'])
        params = optax.apply_updates(state['params'], updates)
        new_state = {
            'params': params,
            'opt_state': opt_state,
            'optimizer': state['optimizer'],
            'step': state['step'] + 1
        }
        return new_state, loss
    
    n_samples = x_data.shape[0]
    steps_per_epoch = n_samples // batch_size
    
    if key is None:
        key = jrandom.PRNGKey(0)
    
    losses = []
    print(f"Training {model_name} with Implicit Score Matching for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        key, subkey = jrandom.split(key)
        perm = jrandom.permutation(subkey, n_samples)
        x_shuffled = x_data[perm]
        v_shuffled = v_data[perm]
        
        epoch_losses = []
        for i in range(steps_per_epoch):
            batch_idx = slice(i * batch_size, (i + 1) * batch_size)
            x_batch = x_shuffled[batch_idx]
            v_batch = v_shuffled[batch_idx]
            key, subkey = jrandom.split(key)
            train_state, loss = train_step(train_state, x_batch, v_batch, subkey)
            epoch_losses.append(loss)
        
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{model_name} - Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    training_time = time.time() - start_time
    print(f"{model_name} training completed in {training_time:.2f} seconds")
    
    return train_state, losses

def evaluate_model(model, params, x_data, v_data, true_scores):
    """Evaluate model performance."""
    predicted_scores = jax.vmap(lambda x, v: model.apply(params, x, v))(x_data, v_data)
    mse = jnp.mean(jnp.sum(jnp.square(predicted_scores - true_scores), axis=-1))
    component_mse = jnp.mean(jnp.square(predicted_scores - true_scores), axis=0)
    return {
        'mse': mse,
        'component_mse': component_mse,
        'predictions': predicted_scores
    }

def plot_training_curves(mlp_losses, resnet_losses, titles, config_name):
    """Plot training loss curves."""
    plots_dir = ensure_plots_dir_exists()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
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

def print_evaluation_results(results, model_name):
    """Print evaluation results."""
    print(f"\n--- {model_name} Evaluation Results ---")
    print(f"Overall MSE: {results['mse']:.6f}")
    print("Component-wise MSE:")
    for i, mse in enumerate(results['component_mse']):
        print(f"  Component {i+1}: {mse:.6f}")
    print()

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
    key1, key2 = jrandom.split(jrandom.PRNGKey(config["random_seed"]))
    X, V = density.sample(key1, size=config["num_particles"])
    true_scores = density.score(X, V)
    
    print(f"Input shapes - X: {X.shape}, V: {V.shape}")
    print(f"True scores shape: {true_scores.shape}")
    
    mlp_model = create_mlp_score_model(
        hidden_dims=config["hidden_dims"],
        activation=config["activation"],
        output_dim=dv
    )
    
    resnet_model = create_resnet_score_model(
        hidden_dims=config["hidden_dims"],
        activation=config["activation"],
        output_dim=dv,
        num_blocks=config["num_blocks"]
    )
    
    key1, key2, key3, key4 = jrandom.split(key2, 4)
    
    mlp_state_explicit = create_train_state(mlp_model, config["learning_rate"], key1, X[0:1], V[0:1])
    resnet_state_explicit = create_train_state(resnet_model, config["learning_rate"], key2, X[0:1], V[0:1])
    mlp_state_implicit = create_train_state(mlp_model, config["learning_rate"], key3, X[0:1], V[0:1])
    resnet_state_implicit = create_train_state(resnet_model, config["learning_rate"], key4, X[0:1], V[0:1])
    
    mlp_output_before = jax.vmap(lambda x, v: mlp_model.apply(mlp_state_explicit['params'], x, v))(X, V)
    resnet_output_before = jax.vmap(lambda x, v: resnet_model.apply(resnet_state_explicit['params'], x, v))(X, V)
    
    print("Evaluating models before training...")
    mlp_results_before = evaluate_model(mlp_model, mlp_state_explicit['params'], X, V, true_scores)
    resnet_results_before = evaluate_model(resnet_model, resnet_state_explicit['params'], X, V, true_scores)
    
    print_evaluation_results(mlp_results_before, "MLP Before Training")
    print_evaluation_results(resnet_results_before, "ResNet Before Training")
    
    print("\n=== Training with Explicit Score Matching ===")
    mlp_state_explicit, mlp_explicit_losses = train_explicit_score_matching(
        mlp_model, mlp_state_explicit, X, V, true_scores, 
        num_epochs=config["num_epochs"], batch_size=config["batch_size"],
        model_name="MLP"
    )
    
    resnet_state_explicit, resnet_explicit_losses = train_explicit_score_matching(
        resnet_model, resnet_state_explicit, X, V, true_scores,
        num_epochs=config["num_epochs"], batch_size=config["batch_size"],
        model_name="ResNet"
    )
    
    print("\n=== Training with Implicit Score Matching ===")
    mlp_state_implicit, mlp_implicit_losses = train_implicit_score_matching(
        mlp_model, mlp_state_implicit, X, V,
        num_epochs=config["num_epochs"], batch_size=config["batch_size"],
        div_mode=config["div_mode"],
        model_name="MLP"
    )
    
    resnet_state_implicit, resnet_implicit_losses = train_implicit_score_matching(
        resnet_model, resnet_state_implicit, X, V,
        num_epochs=config["num_epochs"], batch_size=config["batch_size"],
        div_mode=config["div_mode"],
        model_name="ResNet"
    )
    
    mlp_output_explicit = jax.vmap(lambda x, v: mlp_model.apply(mlp_state_explicit['params'], x, v))(X, V)
    resnet_output_explicit = jax.vmap(lambda x, v: resnet_model.apply(resnet_state_explicit['params'], x, v))(X, V)
    
    mlp_output_implicit = jax.vmap(lambda x, v: mlp_model.apply(mlp_state_implicit['params'], x, v))(X, V)
    resnet_output_implicit = jax.vmap(lambda x, v: resnet_model.apply(resnet_state_implicit['params'], x, v))(X, V)
    
    print("\nEvaluating models after training...")
    
    mlp_results_explicit = evaluate_model(mlp_model, mlp_state_explicit['params'], X, V, true_scores)
    resnet_results_explicit = evaluate_model(resnet_model, resnet_state_explicit['params'], X, V, true_scores)
    
    mlp_results_implicit = evaluate_model(mlp_model, mlp_state_implicit['params'], X, V, true_scores)
    resnet_results_implicit = evaluate_model(resnet_model, resnet_state_implicit['params'], X, V, true_scores)
    
    print_evaluation_results(mlp_results_explicit, "MLP with Explicit Score Matching")
    print_evaluation_results(resnet_results_explicit, "ResNet with Explicit Score Matching")
    print_evaluation_results(mlp_results_implicit, "MLP with Implicit Score Matching")
    print_evaluation_results(resnet_results_implicit, "ResNet with Implicit Score Matching")
    
    # Add run_id to the plot filenames
    plot_training_curves(
        [mlp_explicit_losses, mlp_implicit_losses],
        [resnet_explicit_losses, resnet_implicit_losses],
        ["Explicit Score Matching", "Implicit Score Matching"],
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
            'resnet_implicit': resnet_implicit_losses
        },
        'metrics': {
            'mlp_explicit': mlp_results_explicit,
            'resnet_explicit': resnet_results_explicit,
            'mlp_implicit': mlp_results_implicit,
            'resnet_implicit': resnet_results_implicit
        }
    }
    
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
