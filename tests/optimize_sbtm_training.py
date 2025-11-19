#%%
import os
import time

import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from src import path, utils, loss

from flax import nnx
import optax
from src import score_model

# Show full JAX stack traces (disable filtering)
jax.config.update("jax_traceback_filtering", "off")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
jax.config.update("jax_enable_x64", True)

# Parameters
seed = 42
q = 1
dx = 1
dv = 2
alpha = 0.1
k = 0.5
L = 2 * jnp.pi / k
n = 1000_00
M = 100
dt = 0.02
eta = L / M
cells = (jnp.arange(M) + 0.5) * eta
w = q * L / n
C = 0.1
gamma = -dv

key_v, key_x = jr.split(jr.PRNGKey(seed), 2)
v = jr.normal(key_v, (n, dv))
v = v - jnp.mean(v, axis=0)

def spatial_density(x):
        return (1 + alpha * jnp.cos(k * x)) / (2 * jnp.pi / k)

max_value = jnp.max(spatial_density(cells))
domain = (0.0, float(L))
x = utils.rejection_sample(key_x, spatial_density, domain, max_value=max_value, num_samples=n)

#%%
# -------------------- losses --------------------
def mse_loss(model, batch):
    x, v, s = batch
    pred = model(x, v)
    return loss.mse(pred, s)

def ism_loss(model, batch, key):
    x, v = batch
    return loss.implicit_score_matching_loss(model, x, v, key=key)

# -------------------- single steps --------------------
@nnx.jit
def supervised_step(model, optimizer, batch):
    loss_val, grads = nnx.value_and_grad(mse_loss)(model, batch)
    optimizer.update(grads)
    return loss_val

@nnx.jit
def score_step(model, optimizer, batch, key):
    loss_val, grads = nnx.value_and_grad(ism_loss)(model, batch, key)
    optimizer.update(grads)
    return loss_val

# -------------------- training loops --------------------
def train_initial_model(model, x, v, score, batch_size, num_epochs, abs_tol, lr, verbose=False):
    optimizer = nnx.Optimizer(model, optax.adamw(lr))

    n = x.shape[0]
    for epoch in range(num_epochs):
        full_loss = mse_loss(model, (x, v, score))
        if verbose:
            print(f"Epoch {epoch}: loss = {full_loss:.5f}")
        if full_loss < abs_tol:
            if verbose:
                print(f"Stopping at epoch {epoch} with loss {full_loss:.5f} < {abs_tol}")
            break

        key = jr.PRNGKey(epoch)
        perm = jr.permutation(key, n)
        x_sh, v_sh, s_sh = x[perm], v[perm], score[perm]

        for i in range(0, n, batch_size):
            batch = (x_sh[i:i+batch_size],
                     v_sh[i:i+batch_size],
                     s_sh[i:i+batch_size])
            supervised_step(model, optimizer, batch)

def train_score_model(model, optimizer, x, v, key, batch_size, num_batch_steps, **kwargs):
    n = x.shape[0]
    losses = []
    batch_count = 0

    pbar = tqdm(total=num_batch_steps, desc="Score model training", ncols=100)
    while batch_count < num_batch_steps:
        key, subkey = jr.split(key)
        perm = jr.permutation(subkey, n)
        x = x[perm]
        v = v[perm]

        start = 0
        while start < n and batch_count < num_batch_steps:
            pbar.update(1)
            end = min(start + batch_size, n)
            batch = (x[start:end], v[start:end])
            key, subkey = jr.split(key)
            loss_val = score_step(model, optimizer, batch, subkey)
            losses.append(loss_val)
            start = end
            batch_count += 1

    return losses

#%%
training_config = {
    "batch_size": 2000,
    "num_epochs": 1000, # initial training
    "abs_tol": 1e-3,
    "lr": 2e-4,
    "num_batch_steps": 500,  # at each step
    "train_every": 10   # train every k time steps
}

"Train and save the initial model"
hidden_dims = (256,256)
model = score_model.MLPScoreModel(dx, dv, hidden_dims=hidden_dims)
model_path = os.path.join(path.MODELS, f'landau_damping_dx{dx}_dv{dv}_alpha{alpha}_k{k}/hidden_{str(hidden_dims)}/epochs_{training_config["num_epochs"]}')
model.load(model_path)

#%%
optimizer = nnx.Optimizer(model, optax.adamw(training_config["lr"]))
train_score_model(model, optimizer, x, v, jr.PRNGKey(0), **training_config)
train_score_model(model, optimizer, x, v, jr.PRNGKey(0), **training_config);

