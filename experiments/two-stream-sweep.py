# Two-stream Vlasov PIC with flags for init and time stepping (momentum study)

import argparse
import os
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import wandb

from src import path, utils

# -----------------------------------------------------------------------------


def compute_vortex_shift(x_np, k, L):
    """
    x_np: numpy array of particle positions in [0, L)
    returns (x_mode, shift) where:
      x_mode  = location of dominant k-mode in x in [0, L)
      shift   = signed distance from L/2 in (-L/2, L/2]
    """
    theta = k * x_np
    z = np.exp(1j * theta)
    m = z.mean()
    phase = np.angle(m)          # in (-pi, pi]
    shift = phase / k            # in (-L/2, L/2]
    x_mode = (L / 2 + shift) % L
    return float(x_mode), float(shift)


def parse_args():
    p = argparse.ArgumentParser(description="Two-stream Vlasov PIC with momentum flags")
    p.add_argument("--n", type=int, default=10**6)
    p.add_argument("--M", type=int, default=100)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--dv", type=int, default=2)
    p.add_argument("--final_time", type=float, default=50.0)
    p.add_argument("--alpha", type=float, default=1 / 200)
    p.add_argument("--k", type=float, default=1 / 5)
    p.add_argument("--c", type=float, default=2.4)
    p.add_argument("--seed", type=int, default=42)

    # init flags
    p.add_argument("--init_v_mirror", action="store_true")      # mirror velocities
    p.add_argument("--init_x_symmetric", action="store_true")   # x ~ f, L-x

    # time-stepping flags
    p.add_argument("--step_x_use_vnew", action="store_true")    # x ← x + dt*v_new
    p.add_argument("--step_v_use_Enew", action="store_true")    # v uses E(x_mid) vs E(x)
    p.add_argument("--step_E_use_xnew", action="store_true")    # E uses x_new vs x
    p.add_argument("--step_E_use_vnew", action="store_true")    # E uses v_new vs v
    p.add_argument("--step_E_zero_mean", action="store_true")   # E ← E - mean(E)

    p.add_argument("--wandb_project", type=str, default="vlasov_two_stream_flags")
    p.add_argument("--wandb_run_name", type=str, default="two_stream_flags")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


# -----------------------------------------------------------------------------


def init_two_stream_velocities(key_v, n, dv, c, mirror):
    assert n % 2 == 0
    n_half = n // 2

    if mirror:
        v1 = jr.normal(key_v, (n_half, dv))
        v1 = v1 - v1.mean(axis=0)
        v1 = v1.at[:, 0].add(c)
        v2 = -v1
        return jnp.vstack([v1, v2])

    # non-mirror: independent sampling for each half
    key1, key2 = jr.split(key_v)
    e1 = jnp.array([1.0, 0.0]) if dv == 2 else jnp.concatenate([jnp.array([1.0]), jnp.zeros(dv - 1)])
    v1 = jr.normal(key1, (n_half, dv)) + c * e1
    v2 = jr.normal(key2, (n_half, dv)) - c * e1
    return jnp.vstack([v1, v2])


def init_positions(key_x, n, cells, L, alpha, k, symmetric: bool):
    def spatial_density(x):
        return (1 + alpha * jnp.cos(k * x)) / (2 * jnp.pi / k)

    max_value = jnp.max(spatial_density(cells))
    domain = (0.0, float(L))

    if not symmetric:
        x = utils.rejection_sample(key_x, spatial_density, domain, max_value=max_value, num_samples=n)
        return x

    x1 = utils.rejection_sample(key_x, spatial_density, domain, max_value=max_value, num_samples=n // 2)
    x2 = L - x1
    x = jnp.concatenate([x1, x2])
    return x


def step_vlasov(
    x,
    v,
    E,
    cells,
    eta,
    w,
    dt,
    L,
    key_step,
    step_x_use_vnew,
    step_v_use_Enew,
    step_E_use_xnew,
    step_E_use_vnew,
):
    x_mid = jnp.mod(x + dt * v[:, 0], L)

    if step_v_use_Enew:
        E_part = utils.evaluate_field_at_particles(E, x_mid, cells, eta)
    else:
        E_part = utils.evaluate_field_at_particles(E, x, cells, eta)

    v_new = v.at[:, 0].add(dt * E_part)

    if step_x_use_vnew:
        x_new = jnp.mod(x + dt * v_new[:, 0], L)
    else:
        x_new = jnp.mod(x + dt * v[:, 0], L)

    x_dep = x_new if step_E_use_xnew else x
    v_dep = v_new if step_E_use_vnew else v

    E_new = utils.update_electric_field(E, x_dep, v_dep, cells, eta, w, dt)
    return x_new, v_new, E_new


# -----------------------------------------------------------------------------


def main():
    args = parse_args()
    assert args.n % 2 == 0

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    jax.config.update("jax_enable_x64", not args.fp32)

    wandb_run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )

    n = args.n
    M = args.M
    dt = args.dt
    dv = args.dv
    alpha = args.alpha
    k = args.k
    c = args.c

    def TF(x):
        return "T" if x else "F"

    title = (
        f"Two-stream n={n:.0e} M={M} dt={dt} "
        f"sv={TF(args.init_v_mirror)} "
        f"sx={TF(args.init_x_symmetric)} "
        f"xvn={TF(args.step_x_use_vnew)} "
        f"vEn={TF(args.step_v_use_Enew)} "
        f"Exn={TF(args.step_E_use_xnew)} "
        f"Evn={TF(args.step_E_use_vnew)} "
        f"E0m={TF(args.step_E_zero_mean)}"
    )

    fname_ps = (
        f"ps_n{n:.0e}_M{M}_dt{dt}_"
        f"sv{TF(args.init_v_mirror)}_"
        f"sx{TF(args.init_x_symmetric)}_"
        f"xvn{TF(args.step_x_use_vnew)}_"
        f"vEn{TF(args.step_v_use_Enew)}_"
        f"Exn{TF(args.step_E_use_xnew)}_"
        f"Evn{TF(args.step_E_use_vnew)}_"
        f"E0m{TF(args.step_E_zero_mean)}.png"
    )

    L = 2 * jnp.pi / k
    eta = L / M
    cells = (jnp.arange(M) + 0.5) * eta
    w = L / n

    key = jr.PRNGKey(args.seed)
    key_v, key_x, key_perm_x, key_perm_v = jr.split(key, 4)

    v = init_two_stream_velocities(key_v, n, dv, c, mirror=args.init_v_mirror)
    x = init_positions(key_x, n, cells, L, alpha, k, symmetric=args.init_x_symmetric)

    perm_x = jr.permutation(key_perm_x, n)
    perm_v = jr.permutation(key_perm_v, n)
    x = x[perm_x]
    v = v[perm_v]

    rho = utils.evaluate_charge_density(x, cells, eta, w)
    E = jnp.cumsum(rho - 1.0) * eta
    E = E - jnp.mean(E)

    final_steps = int(args.final_time / dt)
    snapshot_times = np.linspace(0.0, args.final_time, 6)
    snapshot_steps = set(int(round(T / dt)) for T in snapshot_times)

    x_traj, v_traj, t_traj = [], [], []
    E_L2 = [float(jnp.sqrt(jnp.sum(E**2) * eta))]
    mean_v_hist = [float(v[:, 0].mean())]

    start_time = time.perf_counter()
    for istep in tqdm(range(final_steps + 1)):

        if istep in snapshot_steps:
            x_traj.append(np.asarray(x.block_until_ready()))
            v_traj.append(np.asarray(v.block_until_ready()))
            t_traj.append(istep * dt)

        x_mean_centered = float(x.mean() - L / 2)
        v1_mean = float(v[:, 0].mean())

        wandb.log(
            {
                "step": istep,
                "time": istep * dt,
                "x_mean_centered": x_mean_centered,
                "v1_mean": v1_mean,
            },
            step=istep,
        )

        if istep == final_steps:
            break

        key, key_step = jr.split(key)
        x, v, E = step_vlasov(
            x,
            v,
            E,
            cells,
            eta,
            w,
            dt,
            L,
            key_step,
            args.step_x_use_vnew,
            args.step_v_use_Enew,
            args.step_E_use_xnew,
            args.step_E_use_vnew,
        )

        if args.step_E_zero_mean:
            E = E - jnp.mean(E)

        E_norm = float(jnp.sqrt(jnp.sum(E**2) * eta))
        E_L2.append(E_norm)
        mean_v_hist.append(float(v[:, 0].mean()))

        if (istep + 1) % args.log_every == 0:
            elapsed = time.perf_counter() - start_time
            steps_per_sec = (istep + 1) / elapsed
            wandb.log(
                {
                    "step": istep + 1,
                    "time": (istep + 1) * dt,
                    "E_L2": E_norm,
                    "mean_v": mean_v_hist[-1],
                    "steps_per_sec": steps_per_sec,
                },
                step=istep + 1,
            )

    x_final = x_traj[-1]
    v_final = v_traj[-1]

    x_mode, shift = compute_vortex_shift(x_final, k=float(k), L=float(L))
    v1_final = float(v_final[:, 0].mean())

    wandb.log(
        {
            "vortex_center_x": x_mode,
            "vortex_shift": shift,
            "v1_mean_final": v1_final,
        },
        step=final_steps + 1,
    )

    outdir_ps = os.path.join(path.DATA, "plots", "phase_space", "two_stream_flags")
    os.makedirs(outdir_ps, exist_ok=True)

    fig_ps, path_ps = utils.plot_phase_space_snapshots(
        x_traj, v_traj, t_traj, L, title, outdir_ps, fname_ps
    )
    wandb.log({"phase_space_snapshots": wandb.Image(fig_ps)}, step=final_steps + 1)
    plt.close(fig_ps)

    wandb_run.finish()


if __name__ == "__main__":
    main()
