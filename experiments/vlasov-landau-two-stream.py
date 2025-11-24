# Two-stream instability with CLI + wandb logging
# Run with `python experiments/vlasov-landau-two-stream.py --n 1000_000 --M 100 --dt 0.05 --gpu 0 --dv 2 --final_time 50.0 --C 0.08 --alpha 1/200 --score_method kde --wandb_run_name "n1e6_M100_dt0.05_C0.08_kde"`

import argparse
import os
import time

import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import numpy as np

from flax import nnx
import optax

from src import path, utils, score_model


def init_two_stream_velocities(key_v, n, dv, c):
    assert n % 2 == 0, "n must be even"
    n_half = n // 2
    v1 = jr.normal(key_v, (n_half, dv))
    v1 -= jnp.mean(v1, axis=0)
    v1 = v1.at[:, 0].add(c)
    v2 = -v1
    return jnp.vstack([v1, v2])


#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="1D-dvV two-stream Vlasov–Landau PIC solver")
    p.add_argument("--n", type=int, default=10**6, help="Number of particles (must be even)")
    p.add_argument("--M", type=int, default=100, help="Number of spatial cells")
    p.add_argument("--dt", type=float, default=0.05, help="Time step")
    p.add_argument("--gpu", type=int, default=0, help="CUDA_VISIBLE_DEVICES index")
    p.add_argument("--fp32", action="store_true", help="Use float32 instead of float64")
    p.add_argument("--dv", type=int, default=2, help="Velocity dimension")
    p.add_argument("--final_time", type=float, default=50.0, help="Final simulation time")
    p.add_argument("--C", type=float, default=0.08, help="Collision strength (C=0 => collisionless)")
    p.add_argument("--alpha", type=float, default=1/200, help="Amplitude of initial density perturbation")
    p.add_argument("--k", type=float, default=1/5, help="Wave number k")
    p.add_argument("--c", type=float, default=2.4, help="Beam speed for two-stream")
    p.add_argument("--score_method", type=str, default="scaled_kde", choices=["kde", "scaled_kde", "sbtm"])
    p.add_argument("--seed", type=int, default=42, help="Random seed for initialization")

    # sbtm-specific args
    p.add_argument("--sbtm_batch_size", type=int, default=20_000)
    p.add_argument("--sbtm_num_epochs", type=int, default=10_000)
    p.add_argument("--sbtm_abs_tol", type=float, default=1e-4)
    p.add_argument("--sbtm_lr", type=float, default=2e-4)
    p.add_argument("--sbtm_num_batch_steps", type=int, default=100)

    p.add_argument("--wandb_project", type=str, default="vlasov_landau_two_stream", help="wandb project name")
    p.add_argument("--wandb_run_name", type=str, default="two_stream", help="wandb run name")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--log_every", type=int, default=1, help="Log every k steps")
    return p.parse_args()


def main():
    args = parse_args()
    assert args.n % 2 == 0, "--n must be even for two-stream initialization"

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    jax.config.update("jax_enable_x64", not args.fp32)

    wandb_run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )

    seed = args.seed
    q = 1.0
    dv = args.dv
    alpha = args.alpha
    k = args.k
    c = args.c
    L = 2 * jnp.pi / k
    n = args.n
    M = args.M
    dt = args.dt
    eta = L / M
    cells = (jnp.arange(M) + 0.5) * eta
    w = q * L / n
    C = args.C
    gamma = -dv
    dx = 1

    key_v, key_x = jr.split(jr.PRNGKey(seed), 2)
    v = init_two_stream_velocities(key_v, n, dv, c)

    def spatial_density(x):
        return (1 + alpha * jnp.cos(k * x)) / (2 * jnp.pi / k)
    max_value = jnp.max(spatial_density(cells))
    domain = (0.0, float(L))
    
    # symmetric initialization in x
    x1 = utils.rejection_sample(key_x, spatial_density, domain, max_value=max_value, num_samples=n//2)
    x2 = L - x1
    x = jnp.concatenate([x1, x2])
    shuffle_keys = jr.split(key_x, 2)
    perm1 = jr.permutation(shuffle_keys[0], n)
    perm2 = jr.permutation(shuffle_keys[1], n)
    x = x[perm1]
    v = v[perm2]

    score_method = args.score_method
    model = None
    optimizer = None
    training_config = None

    def two_stream_score_v(v_in, c_in):
        v1 = v_in[:, 0]
        r = jnp.tanh(c_in * v1)
        mu = jnp.zeros(v_in.shape[1], v_in.dtype).at[0].set(c_in)
        return -v_in + r[:, None] * mu

    if score_method == "kde":
        score_fn = utils.score_kde
    elif score_method == "scaled_kde":
        score_fn = utils.scaled_score_kde
    elif score_method == "sbtm":
        hidden_dims = (256, 256)
        training_config = {
            "batch_size": args.sbtm_batch_size,
            "num_epochs": args.sbtm_num_epochs,
            "abs_tol": args.sbtm_abs_tol,
            "lr": args.sbtm_lr,
            "num_batch_steps": args.sbtm_num_batch_steps,
        }
        model = score_model.MLPScoreModel(dx, dv, hidden_dims=hidden_dims)
        example_name = "two_stream"
        model_path = os.path.join(
            path.MODELS,
            f"{example_name}_dx{dx}_dv{dv}_alpha{alpha}_k{k}_c{c}/hidden_{str(hidden_dims)}/epochs_{training_config['num_epochs']}",
        )
        if os.path.exists(model_path):
            model.load(model_path)
        else:
            utils.train_initial_model(
                model,
                x,
                v,
                two_stream_score_v(v, c),
                batch_size=training_config["batch_size"],
                num_epochs=training_config["num_epochs"],
                abs_tol=training_config["abs_tol"],
                lr=training_config["lr"],
                verbose=True,
            )
            try:
                model.save(model_path)
            except Exception as e:
                print(f"Warning: could not save model to {model_path}: {e}")
            time.sleep(1)
        optimizer = nnx.Optimizer(model, optax.adamw(training_config["lr"]))

        def score_fn(x_in, v_in, cells_in, eta_in):
            return model(x_in, v_in)
    else:
        raise ValueError(f"Unknown score method: {score_method}")

    print(f"Args: {args}")


    rho = utils.evaluate_charge_density(x, cells, eta, w)
    E = jnp.cumsum(rho - 1) * eta

    def v_target(vv):
        return 0.5 * (jax.scipy.stats.norm.pdf(vv, -c, 1.0) + jax.scipy.stats.norm.pdf(vv, c, 1.0))
    fig_init = utils.visualize_initial(x, v[:, 0], cells, E, rho, eta, L, spatial_density, v_target)
    wandb.log({"initial_state": wandb.Image(fig_init)}, step=0)
    plt.show()
    plt.close(fig_init)

    final_time = args.final_time
    num_steps = int(final_time / dt)
    E_L2 = [jnp.sqrt(jnp.sum(E ** 2) * eta)]

    print(
        f"Two-stream with n={n:.0e}, M={M}, dt={dt}, eta={float(eta):.4f}, "
        f"score_method={score_method}, dv={dv}, C={C}, alpha={alpha}, k={k}, c={c}, gpu={args.gpu}, fp32={args.fp32}"
    )

    if score_method == "sbtm":
        s_plot = model(x, v)
    else:
        s_plot = score_fn(x, v, cells, eta)
    s_true = two_stream_score_v(v, c)

    fig_quiver = utils.plot_score_quiver(v, s_plot, s_true, label=score_method)
    wandb.log({"score_quiver": wandb.Image(fig_quiver)}, step=0)
    plt.show()
    plt.close(fig_quiver)

    snapshot_times = np.linspace(0.0, final_time, 6)
    snapshot_steps = set(int(round(T / dt)) for T in snapshot_times)

    x_traj, v_traj, t_traj = [], [], []
    start_time = time.perf_counter()
    for istep in tqdm(range(num_steps + 1)):
        if istep in snapshot_steps:
            x_host = np.asarray(x.block_until_ready())
            v_host = np.asarray(v.block_until_ready())
            x_traj.append(x_host)
            v_traj.append(v_host)
            t_traj.append(istep * dt)

            if score_method == "sbtm":
                s_snap = model(x, v)
            else:
                s_snap = score_fn(x, v, cells, eta)

            fig_quiver_snap = utils.plot_score_quiver_pred(
                v, s_snap, label=f"{score_method}, t={istep * dt:.2f}"
            )
            wandb.log({"score_quiver": wandb.Image(fig_quiver_snap)}, step=istep)
            plt.close(fig_quiver_snap)

        x, v, E = utils.vlasov_step(x, v, E, cells, eta, dt, L, w)

        if C > 0:
            if score_method == "sbtm":
                s = model(x, v)
                key = jr.PRNGKey(istep)
                utils.train_score_model(
                    model,
                    optimizer,
                    x,
                    v,
                    key,
                    batch_size=training_config["batch_size"],
                    num_batch_steps=training_config["num_batch_steps"],
                )
            else:
                s = score_fn(x, v, cells, eta)
            Q = utils.collision(x, v, s, eta, gamma, L, w)
            v = v - dt * C * Q
            entropy_production = jnp.mean(jnp.sum(s * C * Q, axis=1))
        else:
            entropy_production = 0.0

        electric_energy = jnp.sum(E ** 2) * eta # electric energy
        momentum = jnp.mean(v, axis=0)
        kinetic_energy = 0.5 * jnp.mean(jnp.sum(v ** 2, axis=1))
        total_energy = kinetic_energy + electric_energy
        E_norm = jnp.sqrt(electric_energy)

        E_L2.append(E_norm)
        if (istep + 1) % args.log_every == 0:
            elapsed = time.perf_counter() - start_time
            steps_per_sec = (istep + 1) / elapsed
            mom_dict = {f"momentum/{i}": float(m) for i, m in enumerate(momentum)}
            wandb.log(
                {
                    "step": istep + 1,
                    "time": float((istep + 1) * dt),
                    "steps_per_sec": steps_per_sec,
                    "E_L2": float(E_norm),
                    "electric_energy": float(electric_energy),
                    "kinetic_energy": float(kinetic_energy),
                    "total_energy": float(total_energy),
                    "entropy_production": float(entropy_production),
                    **mom_dict,
                },
                step=istep + 1,
            )


    title = fr"Two-stream α={alpha}, k={k}, c={c}, C={C}, n={n:.0e}, M={M}, Δt={dt}, {score_method}"
    outdir_ps = f"data/plots/phase_space/two_stream_1d_{dv}v/"
    fname_ps = f"two_stream_phase_space_n{n:.0e}_M{M}_dt{dt}_dv{dv}_C{C}_alpha{alpha}_k{k}_c{c}.png"

    fig_ps, path_ps = utils.plot_phase_space_snapshots(
        x_traj, v_traj, t_traj, L, title, outdir_ps, fname_ps
    )

    wandb.log({"phase_space_snapshots": wandb.Image(fig_ps)}, step=num_steps + 1)
    plt.show()
    plt.close(fig_ps)

    snap_art = wandb.Artifact(
        name="two_stream_snapshots",
        type="snapshot_data",
    )
    snap_art.add_file(os.path.join(outdir_ps, fname_ps))
    snap_art.metadata = dict(
        n=n,
        M=M,
        dt=dt,
        dv=dv,
        C=C,
        alpha=alpha,
        k=k,
        c=c,
        score_method=score_method,
    )
    snapshots_dir = os.path.join(
        path.DATA,
        "snapshots",
        f"two_stream_n{n:.0e}_M{M}_dt{dt}_{score_method}_dv{dv}_C{C}_alpha{alpha}_k{k}_c{c}",
    )
    os.makedirs(snapshots_dir, exist_ok=True)
    snapshots_raw_path = os.path.join(snapshots_dir, "snapshots_raw.npz")
    np.savez_compressed(
        snapshots_raw_path,
        x_traj=np.array(x_traj, dtype=object),
        v_traj=np.array(v_traj, dtype=object),
        t_traj=np.array(t_traj),
    )
    snap_art.add_file(snapshots_raw_path)
    wandb.log_artifact(snap_art)

    t_grid = jnp.linspace(0, final_time, num_steps + 2)
    t_np = np.asarray(t_grid)
    E_np = np.asarray(E_L2)

    fig_final = plt.figure(figsize=(6, 4))
    plt.plot(t_np, E_np, marker="o", ms=1, label=f"Simulation (C={C})")

    mask = (t_np > 10.0) & (t_np < 25.0)
    t_mask = t_np[mask]
    E_mask = E_np[mask]

    beta_ref = 0.2258
    if len(t_mask) > 1:
        predicted = np.exp(beta_ref * t_mask)
        predicted *= E_np[0] / predicted[0]
        plt.plot(
            t_mask,
            predicted,
            "r--",
            label=fr"$e^{{\beta t}},\ \beta = {beta_ref:.4f}$",
        )

    plt.xlabel("Time")
    plt.ylabel(r"$||E||_{L^2}$")
    plt.title(f"Two-stream: C={C}, n={n:.0e}, Δt={dt}, dv={dv}, α={alpha}, M={M}, k={k}, c={c}")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()

    outdir = f"data/plots/electric_field_norm/two_stream_1d_{dv}v/"
    os.makedirs(outdir, exist_ok=True)
    fname = f"two_stream_n{n:.0e}_M{M}_dt{dt}_{score_method}_dv{dv}_C{C}_alpha{alpha}_k{k}_c{c}.png"
    p = os.path.join(outdir, fname)
    plt.savefig(p)

    wandb.log({"two_stream_growth": wandb.Image(fig_final)}, step=num_steps + 2)
    wandb.save(p)

    plt.show()
    plt.close(fig_final)

    wandb_run.finish()


if __name__ == "__main__":
    main()
