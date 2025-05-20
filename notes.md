# Notes
- f64 precision is ~10 slower but a lot more accurate. Command: `jax.config.update("jax_enable_x64", True)` 
- collision kernel and NN training dominate time stepping
- utilizing the fact that the kernel is compactly supported gives a huge speedup for the collision kernel computation
- gaussian and rademacher divergence approximations are ~30% faster than exact methods and denoised. Rademacher is slightly more accurate than gaussian and denoised
- backward pass takes ~3x longer than forward pass.
- batch_size of 1000 is as fast as 10. 100_000 is slower.
- collision operator takes 0.04s with 100_000 particles and 256 cells.
- 20 mini-batch steps with batch_size <=1000 takes 0.06s, for a (1024,1024) hidden layers NN.