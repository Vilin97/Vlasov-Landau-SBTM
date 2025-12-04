#%%
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import matplotlib.pyplot as plt
from tqdm import tqdm

alpha = 0.1
k = 0.5
L = 2 * jnp.pi / k

def spatial_density(x):
    return (1.0 + alpha * jnp.cos(k * x)) / (2.0 * jnp.pi / k)

def g_on_grid(x):
    rho = spatial_density(x)
    rho_ion = 1.0 / L
    return rho - rho_ion

def E_true(x):
    return (alpha / (2.0 * jnp.pi)) * jnp.sin(k * x)

M = 512
x_grid = jnp.linspace(0.0, L, M, endpoint=False)
g_grid = g_on_grid(x_grid)
E_grid_true = E_true(x_grid)

class Phi(nnx.Module):
    def __init__(self, rngs, hidden=64, dtype=jnp.float32):
        self.dtype = dtype
        self.l1 = nnx.Linear(2, hidden, rngs=rngs, dtype=dtype)
        self.l2 = nnx.Linear(hidden, hidden, rngs=rngs, dtype=dtype)
        self.l3 = nnx.Linear(hidden, 1, rngs=rngs, dtype=dtype)

    def __call__(self, x):
        # x: (N,)
        x = x[..., None]
        z = jnp.concatenate(
            [jnp.cos(2 * jnp.pi * x / L), jnp.sin(2 * jnp.pi * x / L)],
            axis=-1,
        )  # (N, 2)
        h = jax.nn.tanh(self.l1(z))
        h = jax.nn.tanh(self.l2(h))
        out = self.l3(h)
        return out.squeeze(-1)

def phi_apply(model, x):
    return model(x)

def dphi_dx(model, x):
    def phi_scalar(x0):
        return phi_apply(model, x0[None])[0]
    return jax.vmap(jax.grad(phi_scalar))(x)

def poisson_energy(model, x_colloc, g_colloc):
    phi = phi_apply(model, x_colloc)
    phi_mean = jnp.mean(phi)
    phi_tilde = phi - phi_mean
    dphi = dphi_dx(model, x_colloc)
    grad_term = 0.5 * L * jnp.mean(dphi**2)
    rhs_term  = - L * jnp.mean(g_colloc * phi_tilde)
    return grad_term + rhs_term

def mse_E(model, x_eval):
    dphi = dphi_dx(model, x_eval)
    E_pred = -dphi
    return jnp.mean((E_pred - E_true(x_eval))**2)

rngs = nnx.Rngs(0)
model = Phi(rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adamw(1e-3))

@nnx.jit
def train_step(model, optimizer, x_colloc, g_colloc):
    def loss_fn(m):
        return poisson_energy(m, x_colloc, g_colloc)
    loss_val, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss_val

num_steps = 20000
energy_hist = []
mse_hist = []

for t in tqdm(range(num_steps)):
    loss_val = train_step(model, optimizer, x_grid, g_grid)
    if t % 10 == 0:
        energy_hist.append(loss_val)
        mse_hist.append(mse_E(model, x_grid))

E_pred = -dphi_dx(model, x_grid)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].plot(x_grid, E_grid_true)
axs[0, 0].set_title("True E(x)")

steps = jnp.arange(0, num_steps, 10)
axs[0, 1].plot(steps, energy_hist, label="Poisson energy")
axs[0, 1].plot(steps, mse_hist, label="MSE(E_pred, E_true)")
axs[0, 1].set_yscale("log")
axs[0, 1].legend()
axs[0, 1].set_title("Loss curves")

axs[1, 0].plot(x_grid, E_grid_true, label="true")
axs[1, 0].plot(x_grid, E_pred, "--", label="NN")
axs[1, 0].legend()
axs[1, 0].set_title("E(x): true vs NN")

axs[1, 1].plot(x_grid, E_pred - E_grid_true)
axs[1, 1].set_title("E_pred - E_true")

plt.tight_layout()
plt.show()

#%%
"Compare with cumsum-based approximation"
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import matplotlib.pyplot as plt
from tqdm.auto import trange

alpha = 0.1
k = 0.5
L = 2 * jnp.pi / k

def spatial_density(x):
    return (1.0 + alpha * jnp.cos(k * x)) / (2.0 * jnp.pi / k)

def g_on_grid(x):
    rho = spatial_density(x)
    rho_ion = 1.0 / L
    return rho - rho_ion

def E_true(x):
    return (alpha / (2.0 * jnp.pi)) * jnp.sin(k * x)

M = 512
x_grid = jnp.linspace(0.0, L, M, endpoint=False)
dx = L / M
g_grid = g_on_grid(x_grid)
E_grid_true = E_true(x_grid)

# cumsum-based approximation (discrete integral of g)
E_cumsum = jnp.cumsum(g_grid) * dx
E_cumsum = E_cumsum - jnp.mean(E_cumsum)

class Phi(nnx.Module):
    def __init__(self, rngs, hidden=64, dtype=jnp.float32):
        self.dtype = dtype
        self.l1 = nnx.Linear(2, hidden, rngs=rngs, dtype=dtype)
        self.l2 = nnx.Linear(hidden, hidden, rngs=rngs, dtype=dtype)
        self.l3 = nnx.Linear(hidden, 1, rngs=rngs, dtype=dtype)

    def __call__(self, x):
        x = x[..., None]
        z = jnp.concatenate(
            [jnp.cos(2 * jnp.pi * x / L), jnp.sin(2 * jnp.pi * x / L)],
            axis=-1,
        )
        h = jax.nn.tanh(self.l1(z))
        h = jax.nn.tanh(self.l2(h))
        out = self.l3(h)
        return out.squeeze(-1)

def phi_apply(model, x):
    return model(x)

def dphi_dx(model, x):
    def phi_scalar(x0):
        return phi_apply(model, x0[None])[0]
    return jax.vmap(jax.grad(phi_scalar))(x)

def poisson_energy(model, x_colloc, g_colloc):
    phi = phi_apply(model, x_colloc)
    phi_mean = jnp.mean(phi)
    phi_tilde = phi - phi_mean
    dphi = dphi_dx(model, x_colloc)
    grad_term = 0.5 * L * jnp.mean(dphi**2)
    rhs_term  = - L * jnp.mean(g_colloc * phi_tilde)
    return grad_term + rhs_term

def mse_E(model, x_eval):
    dphi = dphi_dx(model, x_eval)
    E_pred = -dphi
    return jnp.mean((E_pred - E_true(x_eval))**2)

rngs = nnx.Rngs(0)
model = Phi(rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adamw(1e-3))

@nnx.jit
def train_step(model, optimizer, x_colloc, g_colloc):
    def loss_fn(m):
        return poisson_energy(m, x_colloc, g_colloc)
    loss_val, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss_val

num_steps = 20000
energy_hist = []
mse_hist = []

for t in trange(num_steps):
    loss_val = train_step(model, optimizer, x_grid, g_grid)
    if t % 10 == 0:
        energy_hist.append(loss_val)
        mse_hist.append(mse_E(model, x_grid))

E_pred = -dphi_dx(model, x_grid)

# EMA for losses
def ema(values, alpha=0.95):
    vals = jnp.array(values)
    ema_vals = [vals[0]]
    for v in vals[1:]:
        ema_vals.append(alpha * ema_vals[-1] + (1 - alpha) * v)
    return jnp.array(ema_vals)

steps = jnp.arange(0, num_steps, 10)
ema_energy = ema(energy_hist, alpha=0.95)
ema_mse = ema(mse_hist, alpha=0.95)

# MSEs vs true field
mse_nn = mse_E(model, x_grid)
mse_cumsum = jnp.mean((E_cumsum - E_grid_true) ** 2)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# 1st subplot: NN field only
axs[0, 0].plot(x_grid, E_pred, label="NN E(x)")
axs[0, 0].set_title("NN E(x)")
axs[0, 0].set_xlabel("x")
axs[0, 0].legend()

# 2nd subplot: EMA losses
axs[0, 1].plot(steps, ema_energy, label="EMA Poisson energy")
axs[0, 1].plot(steps, ema_mse, label="EMA MSE(E_NN, E_true)")
axs[0, 1].set_yscale("log")
axs[0, 1].legend()
axs[0, 1].set_title("Loss curves (EMA)")
axs[0, 1].set_xlabel("step")

# 3rd subplot: NN vs cumsum approximations
axs[1, 0].plot(x_grid, E_pred, label="NN E(x)")
axs[1, 0].plot(x_grid, E_cumsum, "--", label="cumsum E(x)")
axs[1, 0].legend()
axs[1, 0].set_title("E(x) approximations")
axs[1, 0].set_xlabel("x")

# 4th subplot: errors vs true for both
axs[1, 1].plot(x_grid, E_pred - E_grid_true, label="NN E - E_true")
axs[1, 1].plot(x_grid, E_cumsum - E_grid_true, "--", label="cumsum E - E_true")
axs[1, 1].legend()
axs[1, 1].set_title("Errors vs true")
axs[1, 1].set_xlabel("x")

plt.tight_layout()
plt.show()

print("MSE (NN vs true):     ", float(mse_nn))
print("MSE (cumsum vs true): ", float(mse_cumsum))
