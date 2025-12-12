# Train initial score model only (no time stepping)
# Example:
# python experiments/train-weibel-initial.py \
#   --example weibel --n 1000000 --dv 2 --beta 1e-2 --c 0.3 --k 0.2 \
#   --hidden_width 256 --hidden_num 2 --lr 2e-4 --max_epochs 10000 \
#   --batch_size 20000 --abs_tol 1e-4 --gpu 0

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse

import jax
import jax.numpy as jnp
import jax.random as jr
import wandb

from flax import nnx
import optax

from src import utils, score_model


def sample_x_uniform(key_x, n, L):
    return jr.uniform(key_x, (n,), minval=0.0, maxval=L)

def init_weibel_velocities(key_v, n, dv, beta, c):
    assert n % 2 == 0
    n_half = n // 2
    sigma = jnp.sqrt(beta / 2.0)

    k1, k2, k3, k4 = jr.split(key_v, 4)
    v1 = jr.normal(k1, (n,)) * sigma

    v2_1 = jr.normal(k2, (n_half,)) * sigma + c
    v2_2 = jr.normal(k3, (n_half,)) * sigma - c
    v2 = jnp.concatenate([v2_1, v2_2])

    if dv == 2:
        v = jnp.stack([v1, v2], axis=1)
    else:
        v3 = jr.normal(k4, (n,)) * sigma
        v = jnp.stack([v1, v2, v3], axis=1)

    return v - jnp.mean(v, axis=0)


def weibel_score_v(v, beta, c):
    v1 = v[:, 0]
    v2 = v[:, 1]

    s1 = -2.0 * v1 / beta

    e1 = jnp.exp(-(v2 - c) ** 2 / beta)
    e2 = jnp.exp(-(v2 + c) ** 2 / beta)
    num = (v2 - c) * e1 + (v2 + c) * e2
    den = e1 + e2
    s2 = -2.0 / beta * (num / den)

    if v.shape[1] == 2:
        return jnp.stack([s1, s2], axis=1)
    else:
        v3 = v[:, 2]
        s3 = -2.0 * v3 / beta
        return jnp.stack([s1, s2, s3], axis=1)


def parse_args():
    p = argparse.ArgumentParser(description="Train initial score model (analytic supervision)")
    p.add_argument("--example", type=str, default="weibel", choices=["weibel", "two_stream"])
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--dv", type=int, default=3)
    p.add_argument("--beta", type=float, default=1e-2)   # for weibel
    p.add_argument("--c", type=float, default=0.3)
    p.add_argument("--k", type=float, default=0.2)       # for domain length, if needed
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--hidden_width", type=int, default=256)
    p.add_argument("--hidden_num", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_epochs", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=20000)
    p.add_argument("--abs_tol", type=float, default=1e-4)

    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--fp32", action="store_true")

    p.add_argument("--wandb_project", type=str, default="weibel_initial")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return p.parse_args()


def main():
    args = parse_args()

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    jax.config.update("jax_enable_x64", not args.fp32)

    if args.wandb_run_name is None:
        args.wandb_run_name = (
            f"{args.example}_n{args.n}_dv{args.dv}_hw{args.hidden_width}"
            f"_hn{args.hidden_num}_lr{args.lr}"
        )

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )

    key = jr.PRNGKey(args.seed)
    key_x, key_v = jr.split(key, 2)

    L = 2.0 * jnp.pi / args.k
    x = sample_x_uniform(key_x, args.n, L)
    v = init_weibel_velocities(key_v, args.n, args.dv, args.beta, args.c)
    score_true = weibel_score_v(v, args.beta, args.c)

    hidden_dims = tuple([args.hidden_width] * args.hidden_num)
    dx = 1
    model = score_model.MLPScoreModel(dx, args.dv, hidden_dims=hidden_dims)

    
    
    # train
    batch_size=args.batch_size
    num_epochs=args.max_epochs
    abs_tol=args.abs_tol
    lr=args.lr
    verbose=True
    print_every=100

    optimizer = nnx.Optimizer(model, optax.adamw(lr))
    n = x.shape[0]
    full_loss_hist = []
    for epoch in range(num_epochs):
        full_loss = utils.mse_loss(model, (x, v, score_true))
        full_loss_hist.append(full_loss)
        wandb.log({"train/full_loss": float(full_loss)}, step=epoch)
        if verbose and epoch % print_every == 0:
            print(f"Epoch {epoch}: loss = {full_loss:.5f}")
        if full_loss < abs_tol:
            if verbose:
                print(f"Stopping at epoch {epoch} with loss {full_loss:.5f} < {abs_tol}")
            break
        key = jr.PRNGKey(epoch)
        perm = jr.permutation(key, n)
        x_sh, v_sh, s_sh = x[perm], v[perm], score_true[perm]
        for i in range(0, n, batch_size):
            batch = (
                x_sh[i:i+batch_size],
                v_sh[i:i+batch_size],
                s_sh[i:i+batch_size],
            )
            utils.supervised_step(model, optimizer, batch)
    
    run.finish()


if __name__ == "__main__":
    main()
