# Notes
- f64 precision is ~10 slower but a lot more accurate. Command: `jax.config.update("jax_enable_x64", True)` 
- collision kernel and NN training dominate time stepping
- utilizing the fact that the kernel is compactly supported gives a huge speedup for the collision kernel computation
- the collision kernel is a convolution, so we can use FFTs to speed it up
- gaussian and rademacher divergence approximations are ~30% faster than exact methods and denoised