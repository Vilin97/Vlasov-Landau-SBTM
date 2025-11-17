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

from src import path, utils

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
    p.add_argument("--score_method", type=str, default="scaled_kde", choices=["kde", "scaled_kde"])

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

    # Parameters
    seed = 42
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

    key_v, key_x = jr.split(jr.PRNGKey(seed), 2)
    
    v = init_two_stream_velocities(key_v, n, dv, c)

    score_method = args.score_method
    if score_method == "kde":
        score_fn = utils.score_kde
    elif score_method == "scaled_kde":
        score_fn = utils.scaled_score_kde
    else:
        raise ValueError(f"Unknown score method: {score_method}")

    print(f"Args: {args}")

    def spatial_density(x):
        return (1 + alpha * jnp.cos(k * x)) / (2 * jnp.pi / k)

    # Ideal target in v: symmetric two-stream mixture
    def v_target(vv):
        return 0.5 * (
            jax.scipy.stats.norm.pdf(vv, -c, 1.0) +
            jax.scipy.stats.norm.pdf(vv,  c, 1.0)
        )

    max_value = jnp.max(spatial_density(cells))
    domain = (0.0, float(L))
    x = utils.rejection_sample(key_x, spatial_density, domain, max_value=max_value, num_samples=n)

    rho = utils.evaluate_charge_density(x, cells, eta, w)
    E = jnp.cumsum(rho - 1) * eta
    E = E - jnp.mean(E)

    fig_init = utils.visualize_initial(x, v[:, 0], cells, E, rho, eta, L, spatial_density, v_target)
    wandb.log({"initial_state": wandb.Image(fig_init)}, step=0)
    plt.show()
    plt.close(fig_init)

    final_time = args.final_time
    num_steps = int(final_time / dt)
    t = 0.0
    E_L2 = [jnp.sqrt(jnp.sum(E ** 2) * eta)]

    print(
        f"Two-stream with n={n:.0e}, M={M}, dt={dt}, eta={float(eta):.4f}, "
        f"score_method={args.score_method}, dv={dv}, C={C}, alpha={alpha}, k={k}, c={c}, gpu={args.gpu}, fp32={args.fp32}"
    )

    # Quiver of scores before time stepping
    s_kde = score_fn(x, v, cells, eta)
    def two_stream_score_v(v, c):
        # v: (n, dv)
        v1 = v[:, 0]                      # (n,)
        r = jnp.tanh(c * v1)              # (n,)
        mu = jnp.zeros(v.shape[1], v.dtype).at[0].set(c)  # (dv,)
        return -v + r[:, None] * mu       # (n, dv)
    s_true = two_stream_score_v(v, c)

    fig_quiver = utils.plot_score_quiver(v, s_kde, s_true, label=score_method)
    wandb.log({"score_quiver": wandb.Image(fig_quiver)}, step=0)
    plt.show()
    plt.close(fig_quiver)

    # Main time loop with steps/sec logging
    snapshot_times = np.linspace(0.0, final_time, 6)        # [0,10,20,30,40,50]
    snapshot_steps = set(int(round(T / dt)) for T in snapshot_times)

    x_traj, v_traj, t_traj = [], [], []
    start_time = time.perf_counter()
    for istep in tqdm(range(num_steps+1)):
        if istep in snapshot_steps:
            x_host = np.asarray(x.block_until_ready())
            v_host = np.asarray(v.block_until_ready())
            x_traj.append(x_host)
            v_traj.append(v_host)
            t_traj.append(istep * dt)

        x, v, E = utils.vlasov_step(x, v, E, cells, eta, dt, L, w)

        if C > 0:
            s = score_fn(x, v, cells, eta)
            Q = utils.collision(x, v, s, eta, gamma, n, L, w)
            v = v - dt * C * Q

        E = E - jnp.mean(E)
        E_norm = jnp.sqrt(jnp.sum(E ** 2) * eta)
        E_L2.append(E_norm)

        # logging
        if (istep + 1) % args.log_every == 0:
            elapsed = time.perf_counter() - start_time
            steps_per_sec = (istep + 1) / elapsed
            wandb.log({
                "step": istep + 1,
                "time": float((istep + 1) * dt),
                "E_L2": float(E_norm),
                "steps_per_sec": steps_per_sec,
            }, step=istep + 1)

    # Phase-space snapshots from x_traj, v_traj
    title = fr"Two-stream α={alpha}, k={k}, c={c}, C={C}, n={n:.0e}, M={M}, Δt={dt}, {score_method}"
    outdir_ps = f"data/plots/phase_space/two_stream_1d_{dv}v/"
    fname_ps = f"two_stream_phase_space_n{n:.0e}_M{M}_dt{dt}_dv{dv}_C{C}_alpha{alpha}_k{k}_c{c}.png"
    
    fig_ps, path_ps = utils.plot_phase_space_snapshots(
        x_traj, v_traj, t_traj, L, title, outdir_ps, fname_ps
    )
    
    wandb.log({"phase_space_snapshots": wandb.Image(fig_ps)}, step=num_steps + 1)
    plt.show()
    plt.close(fig_ps)

    # log snapshots
    snap_art = wandb.Artifact(
        name="two_stream_snapshots",
        type="snapshot_data"
    )
    snap_art.add_file(os.path.join(outdir_ps, fname_ps))
    snap_art.metadata = dict(
        n=n, M=M, dt=dt, dv=dv, C=C,
        alpha=alpha, k=k, c=c, score_method=score_method,
    )
    np.savez_compressed(
        "snapshots_raw.npz",
        x_traj=np.array(x_traj, dtype=object),
        v_traj=np.array(v_traj, dtype=object),
        t_traj=np.array(t_traj),
    )
    snap_art.add_file("snapshots_raw.npz")
    wandb.log_artifact(snap_art)

    # Post-processing: two-stream growth fit
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
        wandb.log({"beta_ref": beta_ref})

    plt.xlabel("Time")
    plt.ylabel(r"$||E||_{L^2}$")
    plt.title(f"Two-stream: C={C}, n={n:.0e}, Δt={dt}, dv={dv}, α={alpha}, M={M}, k={k}, c={c}")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()

    outdir = f"data/plots/electric_field_norm/two_stream_1d_{dv}v/"
    os.makedirs(outdir, exist_ok=True)
    fname = f"two_stream_n{n:.0e}_M{M}_dt{dt}_{score_method}_dv{dv}_C{C}_alpha{alpha}_k{k}_c{c}.png"
    path = os.path.join(outdir, fname)
    plt.savefig(path)

    wandb.log({"two_stream_growth": wandb.Image(fig_final)}, step=num_steps + 2)
    wandb.save(path)

    plt.show()
    plt.close(fig_final)

    wandb_run.finish()


if __name__ == "__main__":
    main()
