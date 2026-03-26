# Vlasov-Landau-SBTM

A neural score-based particle method for the Vlasov-Maxwell-Landau system. We replace the $O(n^2)$-memory blob (KDE) score estimator in the deterministic particle method of [Bailo et al. (2024)](https://arxiv.org/abs/2404.11841) with score-based transport modeling (SBTM), where a neural network is trained on-the-fly via implicit score matching at $O(n)$ cost.

## Setup

Requires Python 3.10+ with a CUDA-capable GPU.

```bash
pip install jax[cuda] flax optax orbax-checkpoint wandb matplotlib scipy
```

## Usage

Three experiments are provided: Landau damping, two-stream instability, and Weibel instability.

```bash
# Landau damping with SBTM (n=1M particles)
python experiments/vlasov-landau-damping.py --n 1000000 --M 100 --dt 0.02 --C 0.4 --score_method sbtm --gpu 0

# Two-stream instability with blob (KDE)
python experiments/vlasov-landau-two-stream.py --n 1000000 --M 100 --dt 0.05 --C 0.32 --score_method blob --gpu 0

# Weibel instability (electromagnetic)
python experiments/vlasov-landau-weibel.py --n 1000000 --M 100 --dt 0.1 --C 0.0001 --dv 3 --score_method sbtm --gpu 0
```

Key arguments:
- `--n`: number of particles
- `--M`: spatial grid cells
- `--dt`: time step
- `--C`: collision strength (0 = collisionless)
- `--score_method`: `sbtm` or `blob`
- `--dv`: velocity dimensions (2 or 3)

Results are logged to [Weights & Biases](https://wandb.ai). Set `--wandb_project` and `--wandb_run_name` to customize.

## Code Structure

| File | Description |
| ---- | ----------- |
| `experiments/vlasov-landau-damping.py` | Landau damping experiment (electrostatic) |
| `experiments/vlasov-landau-two-stream.py` | Two-stream instability experiment (electrostatic) |
| `experiments/vlasov-landau-weibel.py` | Weibel instability experiment (electromagnetic) |
| `src/score_model.py` | MLP and ResNet score network architectures |
| `src/loss.py` | Implicit and explicit score matching losses |
| `src/utils.py` | Training loops and visualization |

## Citation

```bibtex
@article{ilin2025sbtm,
  title={A Neural Score-Based Particle Method for the Vlasov-Maxwell-Landau System},
  author={Ilin, Vasily and Hu, Jingwei},
  year={2026}
}
```
