"""Compare memory and speed of different implementations of evaluate_charge_density."""

#%%
@jax.jit
def psi(x, eta, box_length):
    "psi_eta(x) = max(0, 1-|x|/eta) / eta."
    x = centered_mod(x, box_length)
    kernel = jnp.maximum(0.0, 1.0 - jnp.abs(x / eta))
    return kernel / eta

# memory O(N + m)
@jax.jit
def evaluate_charge_density(x, cells, eta, box_length, qe=1):
    rho = qe * jax.vmap(lambda cell: jnp.mean(psi(x - cell, eta, box_length)))(cells)
    return rho

# memory O(N + m), speed ~5x slower
@jax.jit
def evaluate_charge_density_stream(x, cells, eta, box_length, qe=1):
    def body(rho, x_i):
        rho += psi(x_i - cells, eta, box_length)   # length‑m update
        return rho, None

    rho, _ = jax.lax.scan(body, jnp.zeros_like(cells), x)
    return qe * rho / x.size

# same memory and speed as `evaluate_charge_density`
@jax.jit
def evaluate_charge_density_remat(x, cells, eta, L, qe=1):
    rho = qe * jax.remat(
        jax.vmap(lambda cell: jnp.mean(psi(x - cell, eta, L)))
    )(cells)
    return rho

#%%
import jax, jax.numpy as jnp, psutil, os

N, m = 1_000_000, 1000
x      = jnp.linspace(0, 1, N)
cells  = jnp.linspace(0, 1, m)
E      = jnp.ones_like(cells)        # shape (m,)
eta    = 0.05
L      = 1.0

proc = psutil.Process(os.getpid())

compiled = evaluate_charge_density.lower(
    x, cells, eta, L
).compile()
print("Mb accessed:", compiled.cost_analysis()["bytes accessed"] / 1e6) # grows like N*m
before = proc.memory_info().rss
evaluate_charge_density(x, cells, eta, L).block_until_ready()
print((proc.memory_info().rss - before)/1e6, "MB")  # grows like N or m only

compiled = evaluate_charge_density_stream.lower(
    x, cells, eta, L
).compile()
print("bytes accessed:", compiled.cost_analysis()["bytes accessed"]/1e6, "MB") # grows like N*m

before = proc.memory_info().rss
evaluate_charge_density_stream(x, cells, eta, L).block_until_ready()
print("Δ RSS:", (proc.memory_info().rss - before)/1e6, "MB")   # only a few‑tens MB

compiled = evaluate_charge_density_remat.lower(
    x, cells, eta, L
).compile()
print("bytes accessed (remat):", compiled.cost_analysis()["bytes accessed"]/1e6, "MB")

before = proc.memory_info().rss
evaluate_charge_density_remat(x, cells, eta, L).block_until_ready()
print("Δ RSS (remat):", (proc.memory_info().rss - before)/1e6, "MB")

import time
# Time evaluate_charge_density
start = time.time()
evaluate_charge_density(x, cells, eta, L).block_until_ready()
elapsed1 = time.time() - start
print(f"evaluate_charge_density time: {elapsed1:.4f} s")

# Time evaluate_charge_density_stream
start = time.time()
evaluate_charge_density_stream(x, cells, eta, L).block_until_ready()
elapsed2 = time.time() - start
print(f"evaluate_charge_density_stream time: {elapsed2:.4f} s")

# Time evaluate_charge_density_remat
start = time.time()
evaluate_charge_density_remat(x, cells, eta, L).block_until_ready()
elapsed3 = time.time() - start
print(f"evaluate_charge_density_remat time: {elapsed3:.4f} s")

# Check that both functions give the same output
rho1 = evaluate_charge_density(x, cells, eta, L)
rho2 = evaluate_charge_density_stream(x, cells, eta, L)
print("Max abs diff:", jnp.max(jnp.abs(rho1 - rho2)))
assert jnp.allclose(rho1, rho2, atol=1e-6), "Outputs do not match!"