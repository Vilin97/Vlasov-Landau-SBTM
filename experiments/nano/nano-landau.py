#%%
"Reproduce homogeneous SBTM experiments"

import os
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_enable_x64", True)

B = 1/24
#%%
"Landau operator and KDE score"
def A_apply(dv, ds, gamma, eps=1e-14):
    """
    A(dv) · ds : (n,d)
    dv         : (n,d)
    ds         : (n,d)
    O(nd)
    """
    v2   = jnp.sum(dv * dv, axis=-1, keepdims=True) + eps      # ‖dv‖² per row
    vg   = v2 ** (gamma / 2)                                   # ‖dv‖^γ
    dvds = jnp.sum(dv * ds, axis=-1, keepdims=True)            # dv·ds per row
    return vg * (v2 * ds - dvds * dv)

@jax.jit
def collision(v, s, gamma):
    """
    Q_i = 1/n Σ_j A(v_i−v_j)(s_i−s_j)
    Q : (n,d)
    v : (n,d)
    s : (n,d)
    O(n²d)
    """
    n = v.shape[0]                      
    dv = v[:, None, :] - v[None, :, :]  # (n,n,d)
    ds = s[:, None, :] - s[None, :, :]  # (n,n,d)
    A_ds = A_apply(dv, ds, gamma)       # (n,n,d)
    return jnp.sum(A_ds, axis=1) / n    # (n,d)

@jax.jit
def score_kde(v, h=None, eps=1e-12):
    """
    ∇_v log f(v) via Gaussian KDE.
    v : (n, d)
    h : (d,) per-dim bandwidth (Scott's rule if None)
    returns (n, d)
    """
    n, d = v.shape
    if h is None:
        # Scott's rule (per-dimension)
        sigma = jnp.std(v, axis=0, ddof=1) + eps
        h = sigma * n ** (-1.0 / (d + 4.0))  # (d,)

    dv = v[:, None, :] - v[None, :, :]            # (n, n, d)
    exps = -0.5 * jnp.sum((dv / h) ** 2, axis=-1) # (n, n)
    w = jnp.exp(exps)                             # Gaussian weights
    Z = jnp.sum(w, axis=1, keepdims=True) + eps   # (n,1)

    mu = w @ v / Z                                # weighted mean (n,d)
    return (mu - v) / (h ** 2)                    # score (n,d)

def velocity(v, gamma):
    s = score_kde(v)               # (n,d)
    Q = collision(v, s, gamma)     # (n,d)
    return -B * Q                  # (n,d)

@jax.jit
def step(v, gamma, dt):
    return v + dt * velocity(v, gamma)

#%%
def sample_anisotropic(key, n, d):
    "anisotropic Gaussian init: σ1=1.8, σ2=0.2, others 1"
    sigmas = jnp.array([1.8, 0.2] + [1.0] * max(0, d - 2))[:d]
    return jr.normal(key, (n, d)) * jnp.sqrt(sigmas)

@jax.jit
def cov_emp(v):
    "Empirical covariance of v: (1/n) Σ_i (v_i − μ)(v_i − μ)ᵀ"
    mu = jnp.mean(v, axis=0, keepdims=True)
    X = v - mu
    return (X.T @ X) / v.shape[0]

@jax.jit
def entropy_rate(v, gamma):
    "(1/n) Σ_i s_t(X_i) · Q_t(X_i)"
    s = score_kde(v)
    vel = velocity(v, gamma)
    return jnp.sum(s * vel) / v.shape[0]

def run_landau(key, d, n, dt, t_end, log_every, gamma=0.0):
    v = sample_anisotropic(key, n, d)
    sigmas0 = jnp.array([1.8, 0.2] + [1.0] * max(0, d - 2))[:d]

    steps = int(jnp.round(t_end / dt))
    ts, s11_emp, s22_emp, s11_true, s22_true, entr = [], [], [], [], [], []

    t = 0.0
    for k in tqdm(range(steps), desc=f"d={d}, n={n}"):
        v = step(v, gamma, dt)  # uses score_kde + collision
        t += dt
        if (k % log_every) == 0 or k == steps - 1:
            Sigma = cov_emp(v)
            diag_true = cov_true_diag(t, d, sigmas0)
            ts.append(t)
            s11_emp.append(Sigma[0, 0])
            s22_emp.append(Sigma[1, 1])
            s11_true.append(diag_true[0])
            s22_true.append(diag_true[1])
            entr.append(entropy_rate(v, gamma))

    return (jnp.array(ts),
            jnp.array(s11_emp), jnp.array(s11_true),
            jnp.array(s22_emp), jnp.array(s22_true),
            jnp.array(entr))

#%%
"Define losses"
def score_gaussian(v, sigma):
    "Score function of Gaussian N(0, diag(sigma)) at v"
    return -v / sigma

def loss_explicit(model, v, s_true):
    "explicit score matching loss for Gaussian target"
    s_pred = model(v)                      # (n,d)
    return jnp.sum((s_pred - s_true) ** 2) / v.shape[0]

def loss_implicit(model, v, z, alpha=0.05):
    "denoising trick for score matching loss, eq. (3.5) in Ilin, Wang, Hu 2025"
    s_pred = model(v)                      # (n,d)
    s_lo   = model(v - alpha * z)           # (n,d)
    s_hi   = model(v + alpha * z)           # (n,d)
    div_s = jnp.sum((s_hi - s_lo) * z) / alpha  # scalar
    return (jnp.sum(s_pred ** 2) + div_s) / v.shape[0]

# # loss_explicit ~ loss_implicit + |score_gaussian|^2 / n
v = sample_anisotropic(jr.PRNGKey(0), n=10000, d=3)
sigma = jnp.array([1.8, 0.2, 1.0])
s_true = score_gaussian(v, sigma)
print("Loss explicit:", loss_explicit(score_kde, v, s_true))
z = jr.normal(jr.PRNGKey(1), v.shape)
print("Loss implicit:", loss_implicit(score_kde, v, z))
print("difference:", loss_explicit(score_kde, v, s_true) - loss_implicit(score_kde, v, z))
print("Expected diff:", jnp.sum(score_gaussian(v, sigma) ** 2) / v.shape[0])

#%%
"Define model: a small MLP"
from flax import nnx

class MLPScoreModel(nnx.Module):
    """MLP-based implementation of a score model."""
    def __init__(self, input_dim, hidden_dims=[128, 128], activation=nnx.soft_sign, seed=0, dtype=jnp.float64):
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dtype = dtype
        rngs = nnx.Rngs(seed)
        
        # Initialize layers
        self.layers = []
        dim = input_dim
        for hidden_dim in self.hidden_dims:
            self.layers.append(nnx.Linear(dim, hidden_dim, rngs=rngs, dtype=self.dtype))
            dim = hidden_dim
        self.final_layer = nnx.Linear(dim, input_dim, rngs=rngs, dtype=self.dtype)
    
    def __call__(self, v):
        h = v
        for layer in self.layers:
            h = self.activation(layer(h))
        outputs = self.final_layer(h)
        return outputs
d = 3
sigma = jnp.array([1.8, 0.2, 1.0])
v = sample_anisotropic(jr.PRNGKey(0), n=1000, d=d)
model = MLPScoreModel(d)
model(v)

#%%
"Train initial model to learn Gaussian score"
import optax

# train initial model
batch_size = 1024
num_epochs = 1000
abs_tol    = 1e-3
lr         = 1e-3

optimizer  = nnx.Optimizer(model, optax.adamw(lr))
score_vals = score_gaussian(v, sigma)
loss_fn    = lambda model, v_batch, s_batch: loss_explicit(model, v_batch, s_batch)

for epoch in range(num_epochs):
    loss = loss_fn(model, v, score_vals)
    print(f"Epoch {epoch}: loss = {loss:.5f}")
    if loss < abs_tol:
        print(f"Stopping at epoch {epoch} with loss {loss :.5f} < {abs_tol}")
        break

    perm = jax.random.permutation(jax.random.PRNGKey(epoch), len(v))
    v_shuffled, s_shuffled = v[perm], score_vals[perm]
    for i in range(0, len(v), batch_size):
        v_batch = v_shuffled[i:i + batch_size]
        s_batch = s_shuffled[i:i + batch_size]
        loss_value, grads = nnx.value_and_grad(loss_fn)(model, v_batch, s_batch)
        optimizer.update(grads)

# %%
model = MLPScoreModel(d)
model(v)

optimizer  = nnx.Optimizer(model, optax.adamw(lr))
loss_fn    = lambda model, v_batch, z_batch: loss_implicit(model, v_batch, z_batch)
key = jr.PRNGKey(42)
for epoch in range(100):
    s = score_gaussian(v, sigma)
    explicit_loss_val = loss_explicit(model, v, s)
    z = jr.normal(key, v.shape)
    loss_implicit_val = loss_implicit(model, v, z)
    print(f"Epoch {epoch}: explicit loss = {explicit_loss_val:.5f}, implicit loss = {loss_implicit_val:.5f}")

    perm = jax.random.permutation(jax.random.PRNGKey(epoch), len(v))
    v_shuffled, s_shuffled = v[perm], score_vals[perm]
    batch_key = key
    for i in range(0, len(v), batch_size):
        batch_key = jr.fold_in(batch_key, epoch)
        v_batch = v_shuffled[i:i + batch_size]
        z_batch = jr.normal(batch_key, v_batch.shape)
        loss_value, grads = nnx.value_and_grad(loss_fn)(model, v_batch, z_batch)
        optimizer.update(grads)

#%%
v_grid = jnp.linspace(-4, 4, 100).reshape(-1, 1)
v_grid_stacked = jnp.hstack([v_grid, jnp.zeros((len(v_grid), d-1))])
plt.plot(v_grid, score_gaussian(v_grid_stacked, sigma)[:,0], label=f"True score")
plt.plot(v_grid, score_kde(v_grid_stacked)[:,0], label=f"KDE, loss={loss_explicit(score_kde, v, score_gaussian(v, sigma)):.3f}")
plt.plot(v_grid, model(v_grid_stacked)[:,0], label=f"MLP, loss={loss_explicit(model, v, score_gaussian(v, sigma)):.3f}")
plt.legend()
plt.title("Score functions")
plt.show()

#%%
print("Loss explicit:", loss_explicit(model, v, s_true))
print("Loss implicit:", loss_implicit(model, v, jr.PRNGKey(1)))
print("difference:", loss_explicit(model, v, s_true) - loss_implicit(model, v, jr.PRNGKey(1)))
print("Expected diff:", jnp.sum(score_gaussian(v, sigma) ** 2) / v.shape[0])

#%%
"Maxwell -- Example 5.3 from Ilin, Wang, Hu 2025"
@jax.jit
def cov_true_diag(t, d, sigmas0):
    "Σ*(t) = Σ∞ − (Σ∞ − Σ0) e^{−4dt}"
    tr0 = jnp.sum(sigmas0)
    sigma_inf = tr0 / d
    return sigma_inf - (sigma_inf - sigmas0) * jnp.exp(-4.0 * d * t * B)

key = jr.PRNGKey(0)
results = {}
d_values = [2, 3]
n_values = [100, 1_000, 10_000]
for d in d_values:
    for n in n_values:
        results[(n,d)] = run_landau(key, d=d, n=n, dt=0.01, t_end=4.0, log_every=5, gamma=0.0)

#%%
fig, axs = plt.subplots(len(d_values), 3, figsize=(13, 6), sharex='col')

for row, d in enumerate(d_values):
    # plot empirical trajectories for all n on the same subplots
    for i, n in enumerate(n_values):
        ts, s11_e, s11_t, s22_e, s22_t, ent = results[(n, d)]
        axs[row, 0].plot(ts, s11_e, label=f"blob n={n}", alpha=0.9)
        axs[row, 1].plot(ts, s22_e, label=f"blob n={n}", alpha=0.9)
        axs[row, 2].plot(ts, ent, label=f"blob n={n}", alpha=0.9)
        # axs[row, 2].set_yscale('log')

    # plot analytic curves once per subplot (same analytic regardless of n)
    axs[row, 0].plot(ts, s11_t, ls="--", color="k", label="analytic")
    axs[row, 1].plot(ts, s22_t, ls="--", color="k", label="analytic")

    axs[row, 0].set_title(f"d={d}: Σ₁₁(t)")
    axs[row, 0].set_ylabel("Σ₁₁")
    axs[row, 1].set_title(f"d={d}: Σ₂₂(t)")
    axs[row, 1].set_ylabel("Σ₂₂")
    axs[row, 2].set_title(f"Entropy rate (d={d})")
    axs[row, 2].set_ylabel(r"$\frac{1}{n}\sum s\cdot v$")
    axs[row, 2].set_xlabel("t")

# formatting
for ax in axs.flat:
    ax.grid(True, linestyle=":", linewidth=0.5)
axs[1, 0].legend(loc="best")
axs[1, 1].legend(loc="best")
axs[1, 2].legend(loc="best")

plt.suptitle("Maxwell collisions (γ=0)")
plt.tight_layout()
plt.show()
# %%
"Coulomb -- Example 5.4 from Ilin, Wang, Hu 2025"
def cov_true_diag(t, d, sigmas0):
    return jnp.ones(d)

results = {}
d_values = [2, 3]
n_values = [100, 1_000, 10_000]
for d in d_values:
    for n in n_values:
        results[(n,d)] = run_landau(key, d=d, n=n, dt=1, t_end=300, log_every=5, gamma=-3)


#%%
fig, axs = plt.subplots(len(d_values), 3, figsize=(13, 6), sharex='col')

for row, d in enumerate(d_values):
    # plot empirical trajectories for all n on the same subplots
    for i, n in enumerate(n_values):
        ts, s11_e, s11_t, s22_e, s22_t, ent = results[(n, d)]
        axs[row, 0].plot(ts, s11_e, label=f"blob n={n}", alpha=0.9)
        axs[row, 1].plot(ts, s22_e, label=f"blob n={n}", alpha=0.9)
        axs[row, 2].plot(ts, ent, label=f"blob n={n}", alpha=0.9)
        # axs[row, 2].set_yscale('log')

    # plot analytic curves once per subplot (same analytic regardless of n)
    axs[row, 0].plot(ts, s11_t, ls="--", color="k", label="equilibrium")
    axs[row, 1].plot(ts, s22_t, ls="--", color="k", label="equilibrium")

    axs[row, 0].set_title(f"d={d}: Σ₁₁(t)")
    axs[row, 0].set_ylabel("Σ₁₁")
    axs[row, 1].set_title(f"d={d}: Σ₂₂(t)")
    axs[row, 1].set_ylabel("Σ₂₂")
    axs[row, 2].set_title(f"Entropy rate (d={d})")
    axs[row, 2].set_ylabel(r"$\frac{1}{n}\sum s\cdot v$")
    axs[row, 2].set_xlabel("t")

# formatting
for ax in axs.flat:
    ax.grid(True, linestyle=":", linewidth=0.5)
axs[1, 0].legend(loc="best")
axs[1, 1].legend(loc="best")
axs[1, 2].legend(loc="best")

plt.suptitle("Coulomb collisions (γ=−3)")
plt.tight_layout()
plt.show()
