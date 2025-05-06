"Test periodic boundary condition"
import jax.numpy as jnp
from src.solver import centered_mod, psi

L = 1 # domains size
x0 = jnp.array([-0.01])
x1 = jnp.array([0.01])
x2 = jnp.array([0.49])
x3 = jnp.array([0.51])
x4 = jnp.array([0.99])
x5 = jnp.array([1.01])

# test mod
assert jnp.allclose(jnp.mod(x0, L), jnp.mod(x4, L))
assert jnp.allclose(jnp.mod(x1, L), jnp.mod(x5, L))
assert jnp.allclose(jnp.mod(x2, L), jnp.mod(x2, L))

# test centered_mod
assert jnp.allclose(jnp.abs(centered_mod(x0-x1, L)), 0.02)
assert jnp.allclose(jnp.abs(centered_mod(x1-x0, L)), 0.02)
assert jnp.allclose(jnp.abs(centered_mod(x2-x3, L)), 0.02)
assert jnp.allclose(jnp.abs(centered_mod(x0-x2, L)), 0.5)
assert jnp.allclose(jnp.abs(centered_mod(x0-x3, L)), 0.48)
assert jnp.allclose(jnp.abs(centered_mod(x0-x4, L)), 0.)
assert jnp.allclose(jnp.abs(centered_mod(x0-x5, L)), 0.02)

assert jnp.allclose(jnp.abs(centered_mod(x1-x2, L)), 0.48)
assert jnp.allclose(jnp.abs(centered_mod(x1-x3, L)), 0.5)
assert jnp.allclose(jnp.abs(centered_mod(x1-x4, L)), 0.02)
assert jnp.allclose(jnp.abs(centered_mod(x1-x5, L)), 0.)

assert jnp.allclose(jnp.abs(centered_mod(x2-x3, L)), 0.02)
assert jnp.allclose(jnp.abs(centered_mod(x2-x4, L)), 0.5)
assert jnp.allclose(jnp.abs(centered_mod(x2-x5, L)), 0.48)

# test psi
eta = 0.1
assert jnp.allclose(psi(x0 - x0, eta, L) * eta, 1.0)
assert jnp.allclose(psi(x1 - x0, eta, L) * eta, 0.8)