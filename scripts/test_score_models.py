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

# Import your modules
from src.score_model import create_mlp_score_model, create_resnet_score_model
from src.density import CosineNormal
from src.loss import explicit_score_matching_loss, implicit_score_matching_loss

# Configuration variables for easy modification
LEARNING_RATE = 5e-4  # Learning rate for model training
NUM_EPOCHS = 50      # Number of training epochs
BATCH_SIZE = 64       # Batch size for training

# Set random seed for reproducibility
key = jrandom.PRNGKey(42)

# Create plots directory if it doesn't exist
def ensure_plots_dir_exists():
    """Ensure the plots directory exists."""
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir

def create_train_state(model, learning_rate, key, x_sample, v_sample):
    """Create initial training state with model parameters and optimizer."""
    # Initialize model parameters
    params = model.init(key, x_sample, v_sample)
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    return {
        'params': params,
        'opt_state': opt_state,
        'optimizer': optimizer,
        'step': 0
    }

def train_explicit_score_matching(model, train_state, x_data, v_data, true_scores, 
                                  num_epochs=1000, batch_size=32, key=None, model_name="Model"):
    """Train model using explicit score matching loss."""
    
    # Separate the computation of loss and gradients (JIT-compatible part)
    @jax.jit
    def compute_loss_and_grads(params, x_batch, v_batch, score_batch):
        def loss_fn(params):
            # Wrapper for model inference
            def model_fn(x, v):
                return model.apply(params, x, v)
            
            # Compute explicit score matching loss
            loss = explicit_score_matching_loss(model_fn, x_batch, v_batch, score_batch)
            return loss
        
        # Compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(params)
        return loss, grads
    
    # Non-JIT train step that uses the JIT-compiled gradient computation
    def train_step(state, x_batch, v_batch, score_batch):
        # Compute loss and gradients using JIT-compiled function
        loss, grads = compute_loss_and_grads(state['params'], x_batch, v_batch, score_batch)
        
        # Update optimizer state
        updates, opt_state = state['optimizer'].update(grads, state['opt_state'], state['params'])
        
        # Update parameters
        params = optax.apply_updates(state['params'], updates)
        
        # Update training state
        new_state = {
            'params': params,
            'opt_state': opt_state,
            'optimizer': state['optimizer'],
            'step': state['step'] + 1
        }
        
        return new_state, loss
    
    # Number of samples and batches
    n_samples = x_data.shape[0]
    steps_per_epoch = n_samples // batch_size
    
    # Initialize random key and metrics
    if key is None:
        key = jrandom.PRNGKey(0)
    
    losses = []
    
    # Training loop
    print(f"Training {model_name} with Explicit Score Matching for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Shuffle data for each epoch
        key, subkey = jrandom.split(key)
        perm = jrandom.permutation(subkey, n_samples)
        x_shuffled = x_data[perm]
        v_shuffled = v_data[perm]
        scores_shuffled = true_scores[perm]
        
        # Process batches
        epoch_losses = []
        for i in range(steps_per_epoch):
            # Get batch
            batch_idx = slice(i * batch_size, (i + 1) * batch_size)
            x_batch = x_shuffled[batch_idx]
            v_batch = v_shuffled[batch_idx]
            score_batch = scores_shuffled[batch_idx]
            
            # Update model
            train_state, loss = train_step(train_state, x_batch, v_batch, score_batch)
            epoch_losses.append(loss)
        
        # Record metrics
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        losses.append(avg_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{model_name} - Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    training_time = time.time() - start_time
    print(f"{model_name} training completed in {training_time:.2f} seconds")
    
    return train_state, losses

def train_implicit_score_matching(model, train_state, x_data, v_data, 
                                 num_epochs=1000, batch_size=32, key=None, div_mode='reverse', model_name="Model"):
    """Train model using implicit score matching loss."""
    
    # Separate the computation of loss and gradients (JIT-compatible part)
    @jax.jit
    def compute_loss_and_grads(params, x_batch, v_batch, subkey):
        def loss_fn(params):
            # Wrapper for model inference
            def model_fn(x, v):
                return model.apply(params, x, v)
            
            # Compute implicit score matching loss
            loss = implicit_score_matching_loss(
                model_fn, x_batch, v_batch, 
                key=subkey, div_mode=div_mode, n_samples=10
            )
            return loss
        
        # Compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(params)
        return loss, grads
    
    # Non-JIT train step that uses the JIT-compiled gradient computation
    def train_step(state, x_batch, v_batch, subkey):
        # Compute loss and gradients using JIT-compiled function
        loss, grads = compute_loss_and_grads(state['params'], x_batch, v_batch, subkey)
        
        # Update optimizer state
        updates, opt_state = state['optimizer'].update(grads, state['opt_state'], state['params'])
        
        # Update parameters
        params = optax.apply_updates(state['params'], updates)
        
        # Update training state
        new_state = {
            'params': params,
            'opt_state': opt_state,
            'optimizer': state['optimizer'],
            'step': state['step'] + 1
        }
        
        return new_state, loss
    
    # Number of samples and batches
    n_samples = x_data.shape[0]
    steps_per_epoch = n_samples // batch_size
    
    # Initialize random key and metrics
    if key is None:
        key = jrandom.PRNGKey(0)
    
    losses = []
    
    # Training loop
    print(f"Training {model_name} with Implicit Score Matching for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Shuffle data for each epoch
        key, subkey = jrandom.split(key)
        perm = jrandom.permutation(subkey, n_samples)
        x_shuffled = x_data[perm]
        v_shuffled = v_data[perm]
        
        # Process batches
        epoch_losses = []
        for i in range(steps_per_epoch):
            # Get batch
            batch_idx = slice(i * batch_size, (i + 1) * batch_size)
            x_batch = x_shuffled[batch_idx]
            v_batch = v_shuffled[batch_idx]
            
            # Generate random key for this batch
            key, subkey = jrandom.split(key)
            
            # Update model
            train_state, loss = train_step(train_state, x_batch, v_batch, subkey)
            epoch_losses.append(loss)
        
        # Record metrics
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        losses.append(avg_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{model_name} - Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    training_time = time.time() - start_time
    print(f"{model_name} training completed in {training_time:.2f} seconds")
    
    return train_state, losses

def evaluate_model(model, params, x_data, v_data, true_scores):
    """Evaluate model performance."""
    # Apply model to data
    predicted_scores = jax.vmap(lambda x, v: model.apply(params, x, v))(x_data, v_data)
    
    # Compute MSE
    mse = jnp.mean(jnp.sum(jnp.square(predicted_scores - true_scores), axis=-1))
    
    # Compute component-wise MSE
    component_mse = jnp.mean(jnp.square(predicted_scores - true_scores), axis=0)
    
    return {
        'mse': mse,
        'component_mse': component_mse,
        'predictions': predicted_scores
    }

def plot_training_curves(mlp_losses, resnet_losses, titles):
    """Plot training loss curves."""
    # Ensure plots directory exists
    plots_dir = ensure_plots_dir_exists()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, (mlp_loss, resnet_loss, title) in enumerate(zip(mlp_losses, resnet_losses, titles)):
        axes[i].plot(mlp_loss, label='MLP')
        axes[i].plot(resnet_loss, label='ResNet')
        axes[i].set_title(f'{title} Training Loss')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        
        # Only use log scale for explicit score matching (which is always positive)
        if 'Explicit' in title:
            axes[i].set_yscale('log')  # Log scale often helps visualize convergence
        else:
            # Use linear scale for implicit score matching (which can be negative)
            axes[i].set_yscale('linear')
            
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Use os.path.join for platform-independent path construction
    plt.savefig(os.path.join(plots_dir, 'training_curves.png'))
    plt.show()

def plot_score_comparisons(X, V, true_scores, 
                           mlp_before, resnet_before, 
                           mlp_explicit, resnet_explicit,
                           mlp_implicit, resnet_implicit):
    """Plot comparisons of score models before and after training."""
    # Ensure plots directory exists
    plots_dir = ensure_plots_dir_exists()
    
    # Convert to numpy for plotting
    X_np = np.array(X)
    true_scores_np = np.array(true_scores)
    mlp_before_np = np.array(mlp_before)
    resnet_before_np = np.array(resnet_before)
    mlp_explicit_np = np.array(mlp_explicit)
    resnet_explicit_np = np.array(resnet_explicit)
    mlp_implicit_np = np.array(mlp_implicit)
    resnet_implicit_np = np.array(resnet_implicit)
    
    # Create figure
    fig, axes = plt.subplots(4, 2, figsize=(18, 24))
    
    # Sort by x-coordinate for clearer visualization
    sort_idx = np.argsort(X_np.flatten())
    X_sorted = X_np.flatten()[sort_idx]
    
    # Plot first component (v1)
    for i, (title, mlp_data, resnet_data) in enumerate([
        ("Before Training", mlp_before_np, resnet_before_np),
        ("After Explicit SM", mlp_explicit_np, resnet_explicit_np),
        ("After Implicit SM", mlp_implicit_np, resnet_implicit_np)
    ]):
        # MLP comparison
        ax = axes[i, 0]
        true_v1 = true_scores_np[:, 0][sort_idx]
        pred_v1 = mlp_data[:, 0][sort_idx]
        ax.plot(X_sorted, true_v1, 'b-', linewidth=2, label='True Score')
        ax.plot(X_sorted, pred_v1, 'r--', linewidth=2, label='MLP Prediction')
        ax.set_title(f'{title} - MLP - First Component')
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ResNet comparison
        ax = axes[i, 1]
        pred_v1 = resnet_data[:, 0][sort_idx]
        ax.plot(X_sorted, true_v1, 'b-', linewidth=2, label='True Score')
        ax.plot(X_sorted, pred_v1, 'r--', linewidth=2, label='ResNet Prediction')
        ax.set_title(f'{title} - ResNet - First Component')
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Error plots
    ax = axes[3, 0]
    ax.plot(X_sorted, np.abs(true_scores_np[:, 0][sort_idx] - mlp_explicit_np[:, 0][sort_idx]), 'g-', label='Explicit SM')
    ax.plot(X_sorted, np.abs(true_scores_np[:, 0][sort_idx] - mlp_implicit_np[:, 0][sort_idx]), 'r-', label='Implicit SM')
    ax.set_title('MLP Absolute Error - First Component')
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Absolute Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[3, 1]
    ax.plot(X_sorted, np.abs(true_scores_np[:, 0][sort_idx] - resnet_explicit_np[:, 0][sort_idx]), 'g-', label='Explicit SM')
    ax.plot(X_sorted, np.abs(true_scores_np[:, 0][sort_idx] - resnet_implicit_np[:, 0][sort_idx]), 'r-', label='Implicit SM')
    ax.set_title('ResNet Absolute Error - First Component')
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Absolute Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'score_component1_comparison.png'))
    plt.show()
    
    # Create second figure for the second component
    fig, axes = plt.subplots(4, 2, figsize=(18, 24))
    
    # Plot second component (v2)
    for i, (title, mlp_data, resnet_data) in enumerate([
        ("Before Training", mlp_before_np, resnet_before_np),
        ("After Explicit SM", mlp_explicit_np, resnet_explicit_np),
        ("After Implicit SM", mlp_implicit_np, resnet_implicit_np)
    ]):
        # MLP comparison
        ax = axes[i, 0]
        true_v2 = true_scores_np[:, 1][sort_idx]
        pred_v2 = mlp_data[:, 1][sort_idx]
        ax.plot(X_sorted, true_v2, 'b-', linewidth=2, label='True Score')
        ax.plot(X_sorted, pred_v2, 'r--', linewidth=2, label='MLP Prediction')
        ax.set_title(f'{title} - MLP - Second Component')
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ResNet comparison
        ax = axes[i, 1]
        pred_v2 = resnet_data[:, 1][sort_idx]
        ax.plot(X_sorted, true_v2, 'b-', linewidth=2, label='True Score')
        ax.plot(X_sorted, pred_v2, 'r--', linewidth=2, label='ResNet Prediction')
        ax.set_title(f'{title} - ResNet - Second Component')
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Error plots
    ax = axes[3, 0]
    ax.plot(X_sorted, np.abs(true_scores_np[:, 1][sort_idx] - mlp_explicit_np[:, 1][sort_idx]), 'g-', label='Explicit SM')
    ax.plot(X_sorted, np.abs(true_scores_np[:, 1][sort_idx] - mlp_implicit_np[:, 1][sort_idx]), 'r-', label='Implicit SM')
    ax.set_title('MLP Absolute Error - Second Component')
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Absolute Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[3, 1]
    ax.plot(X_sorted, np.abs(true_scores_np[:, 1][sort_idx] - resnet_explicit_np[:, 1][sort_idx]), 'g-', label='Explicit SM')
    ax.plot(X_sorted, np.abs(true_scores_np[:, 1][sort_idx] - resnet_implicit_np[:, 1][sort_idx]), 'r-', label='Implicit SM')
    ax.set_title('ResNet Absolute Error - Second Component')
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Absolute Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'score_component2_comparison.png'))
    plt.show()

def print_evaluation_results(results, model_name):
    """Print evaluation results."""
    print(f"\n--- {model_name} Evaluation Results ---")
    print(f"Overall MSE: {results['mse']:.6f}")
    print("Component-wise MSE:")
    for i, mse in enumerate(results['component_mse']):
        print(f"  Component {i+1}: {mse:.6f}")
    print()

def main():
    # Parameters
    batch_size = 512   # Number of samples
    dx = 1             # Dimension of position (x)
    dv = 2             # Dimension of velocity (v)
    
    # Create sample data - use a larger dataset for training
    density = CosineNormal(alpha=0.4, k=0.5, dx=dx, dv=dv)
    key1, key2 = jrandom.split(key)
    X, V = density.sample(key1, size=batch_size)
    
    # Compute true scores for comparison
    true_scores = density.score(X, V)
    
    print(f"Input shapes - X: {X.shape}, V: {V.shape}")
    print(f"True scores shape: {true_scores.shape}")
    
    # Create models
    mlp_model = create_mlp_score_model(
        hidden_dims=(64, 128, 64),
        activation=nn.swish,
        output_dim=dv
    )
    
    resnet_model = create_resnet_score_model(
        hidden_dims=(64, 128, 64),
        activation=nn.swish,
        output_dim=dv,
        num_blocks=2
    )
    
    # Create training states for different training approaches
    key1, key2, key3, key4 = jrandom.split(key2, 4)
    
    # Initialize states for explicit score matching using the global learning rate
    mlp_state_explicit = create_train_state(mlp_model, LEARNING_RATE, key1, X[0:1], V[0:1])
    resnet_state_explicit = create_train_state(resnet_model, LEARNING_RATE, key2, X[0:1], V[0:1])
    
    # Initialize states for implicit score matching using the global learning rate
    mlp_state_implicit = create_train_state(mlp_model, LEARNING_RATE, key3, X[0:1], V[0:1])
    resnet_state_implicit = create_train_state(resnet_model, LEARNING_RATE, key4, X[0:1], V[0:1])
    
    # Get outputs before training
    mlp_output_before = jax.vmap(lambda x, v: mlp_model.apply(mlp_state_explicit['params'], x, v))(X, V)
    resnet_output_before = jax.vmap(lambda x, v: resnet_model.apply(resnet_state_explicit['params'], x, v))(X, V)
    
    print("Evaluating models before training...")
    mlp_results_before = evaluate_model(mlp_model, mlp_state_explicit['params'], X, V, true_scores)
    resnet_results_before = evaluate_model(resnet_model, resnet_state_explicit['params'], X, V, true_scores)
    
    print_evaluation_results(mlp_results_before, "MLP Before Training")
    print_evaluation_results(resnet_results_before, "ResNet Before Training")
    
    # Use the global config for training
    print(f"\nTraining Configuration:")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    
    # Train with explicit score matching
    print("\n=== Training with Explicit Score Matching ===")
    mlp_state_explicit, mlp_explicit_losses = train_explicit_score_matching(
        mlp_model, mlp_state_explicit, X, V, true_scores, 
        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, key=key1, model_name="MLP"
    )
    
    resnet_state_explicit, resnet_explicit_losses = train_explicit_score_matching(
        resnet_model, resnet_state_explicit, X, V, true_scores,
        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, key=key2, model_name="ResNet"
    )
    
    # Train with implicit score matching
    print("\n=== Training with Implicit Score Matching ===")
    mlp_state_implicit, mlp_implicit_losses = train_implicit_score_matching(
        mlp_model, mlp_state_implicit, X, V,
        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, key=key3, model_name="MLP"
    )
    
    resnet_state_implicit, resnet_implicit_losses = train_implicit_score_matching(
        resnet_model, resnet_state_implicit, X, V,
        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, key=key4, model_name="ResNet"
    )
    
    # Get model outputs after training
    mlp_output_explicit = jax.vmap(lambda x, v: mlp_model.apply(mlp_state_explicit['params'], x, v))(X, V)
    resnet_output_explicit = jax.vmap(lambda x, v: resnet_model.apply(resnet_state_explicit['params'], x, v))(X, V)
    
    mlp_output_implicit = jax.vmap(lambda x, v: mlp_model.apply(mlp_state_implicit['params'], x, v))(X, V)
    resnet_output_implicit = jax.vmap(lambda x, v: resnet_model.apply(resnet_state_implicit['params'], x, v))(X, V)
    
    # Evaluate models after training
    print("\nEvaluating models after training...")
    
    mlp_results_explicit = evaluate_model(mlp_model, mlp_state_explicit['params'], X, V, true_scores)
    resnet_results_explicit = evaluate_model(resnet_model, resnet_state_explicit['params'], X, V, true_scores)
    
    mlp_results_implicit = evaluate_model(mlp_model, mlp_state_implicit['params'], X, V, true_scores)
    resnet_results_implicit = evaluate_model(resnet_model, resnet_state_implicit['params'], X, V, true_scores)
    
    print_evaluation_results(mlp_results_explicit, "MLP with Explicit Score Matching")
    print_evaluation_results(resnet_results_explicit, "ResNet with Explicit Score Matching")
    print_evaluation_results(mlp_results_implicit, "MLP with Implicit Score Matching")
    print_evaluation_results(resnet_results_implicit, "ResNet with Implicit Score Matching")
    
    # Plot training curves
    plot_training_curves(
        [mlp_explicit_losses, mlp_implicit_losses],
        [resnet_explicit_losses, resnet_implicit_losses],
        ["Explicit Score Matching", "Implicit Score Matching"]
    )
    
    # Plot score comparisons
    plot_score_comparisons(
        X, V, true_scores,
        mlp_output_before, resnet_output_before,
        mlp_output_explicit, resnet_output_explicit,
        mlp_output_implicit, resnet_output_implicit
    )
    
    # Return data for further analysis
    return {
        'data': (X, V, true_scores),
        'before_training': (mlp_output_before, resnet_output_before),
        'explicit_sm': (mlp_output_explicit, resnet_output_explicit),
        'implicit_sm': (mlp_output_implicit, resnet_output_implicit),
        'losses': {
            'mlp_explicit': mlp_explicit_losses,
            'resnet_explicit': resnet_explicit_losses,
            'mlp_implicit': mlp_implicit_losses,
            'resnet_implicit': resnet_implicit_losses
        }
    }
    
if __name__ == "__main__":
    results = main()

# %%
