# Backfill existing Weibel runs with corrected plots and l2_dist_gaussian_v2 metric
# This script updates the last N runs with:
# 1. Correct l2_dist_gaussian_v2 (v2 marginal to 1D Gaussian, not full dv-dim)
# 2. Log-scale density plots (v2_at_v1_zero_evolution_log, v2_marginal_evolution_log)
#
# Uses wandb.define_metric() with a custom x-axis to avoid "monotonically increasing step" issue.

import os
import numpy as np
import matplotlib.pyplot as plt
import wandb

from src import utils

# Configuration
PROJECT_NAME = "weibel"
NUM_RUNS_TO_BACKFILL = 20

# Initialize wandb API
api = wandb.Api()

# Get the last N runs from the project
runs = api.runs(f"{PROJECT_NAME}", order="-created_at")
runs_list = list(runs)[:NUM_RUNS_TO_BACKFILL]

print(f"Found {len(runs_list)} runs to backfill")

step = 1300
for run in runs_list:
    print(f"\n{'='*60}")
    print(f"Processing run: {run.name} (id: {run.id})")
    print(f"{'='*60}")

    # Get run config
    config = run.config
    dv = config.get("dv", 3)
    beta = config.get("beta", 0.01)
    c = config.get("c", 0.3)
    final_time = config.get("final_time", 125.0)
    dt = config.get("dt", 0.1)

    # Compute theta for this run
    if dv == 2:
        theta = (beta + c**2) / 2
    else:  # dv == 3
        theta = beta / 2 + c**2 / 3

    print(f"  dv={dv}, beta={beta}, c={c}, theta={theta:.6f}")

    # Try to download the trajectory artifact
    try:
        artifacts = run.logged_artifacts()
        traj_artifact = None
        for artifact in artifacts:
            if "trajectories" in artifact.name:
                traj_artifact = artifact
                break

        if traj_artifact is None:
            print(f"  No trajectory artifact found, skipping...")
            continue

        # Download artifact
        artifact_dir = traj_artifact.download()

        # Find the npz file
        npz_files = [f for f in os.listdir(artifact_dir) if f.endswith('.npz')]
        if not npz_files:
            print(f"  No npz file found in artifact, skipping...")
            continue

        npz_path = os.path.join(artifact_dir, npz_files[0])
        data = np.load(npz_path)

        v_traj = data['v_traj']
        t_traj = data['t_traj']

        print(f"  Loaded trajectories: {len(t_traj)} snapshots, t={list(t_traj)}")

        # Resume the run to add new data
        with wandb.init(
            project=PROJECT_NAME,
            id=run.id,
            resume="must"
        ) as resumed_run:
            final_steps = int(final_time / dt)

            # Define custom metrics with "time" as x-axis (avoids step monotonicity issue)
            wandb.define_metric("time")
            wandb.define_metric("l2_dist_gaussian_v2", step_metric="time")

            # 1. Compute and log l2_dist_gaussian_v2 for each snapshot
            print(f"  Computing L2 distances...")
            for i, (v_snap, t_snap) in enumerate(zip(v_traj, t_traj)):
                l2_dist = utils.compute_l2_distance_to_gaussian(
                    v_snap[:, 1:2], dv=1, theta=theta,
                    bounds_v=[(-0.8, 0.8)], bins=200
                )
                # Log with custom x-axis metric in the same dict
                wandb.log({
                    "time": float(t_snap),
                    "l2_dist_gaussian_v2": float(l2_dist)
                })
                print(f"    t={t_snap:.1f}: L2 = {l2_dist:.6f}")

            # 2. Generate log-scale plots
            print(f"  Generating log-scale plots...")

            # v2 at v1≈0 evolution (log scale)
            fig_v2_at_v1_zero_log = utils.plot_v2_at_v1_zero_evolution(
                list(v_traj), list(t_traj),
                bounds_v=[(-0.01, 0.01), (-0.8, 0.8)],
                bins_per_side=(1, 400),
                theta=theta,
                logy=True,
                title=run.name
            )
            wandb.log({"v2_at_v1_zero_evolution_log": wandb.Image(fig_v2_at_v1_zero_log)}, step=step)
            plt.close(fig_v2_at_v1_zero_log)
            print(f"  Saved: v2_at_v1_zero_evolution_log")

            # v2 marginal evolution (log scale)
            fig_v2_marginal_log = utils.plot_v2_marginal_evolution(
                list(v_traj), list(t_traj),
                bounds_v=[(-0.8, 0.8)],
                bins_per_side=400,
                theta=theta,
                logy=True,
                title=run.name
            )
            wandb.log({"v2_marginal_evolution_log": wandb.Image(fig_v2_marginal_log)}, step=step)
            plt.close(fig_v2_marginal_log)
            print(f"  Saved: v2_marginal_evolution_log")

            # Also regenerate the non-log plots with correct 2D Gaussian normalization
            fig_v2_at_v1_zero = utils.plot_v2_at_v1_zero_evolution(
                list(v_traj), list(t_traj),
                bounds_v=[(-0.01, 0.01), (-0.8, 0.8)],
                bins_per_side=(1, 400),
                theta=theta,
                logy=False,
                title=run.name
            )
            wandb.log({"v2_at_v1_zero_evolution": wandb.Image(fig_v2_at_v1_zero)}, step=step)
            plt.close(fig_v2_at_v1_zero)
            print(f"  Saved: v2_at_v1_zero_evolution")

            fig_v2_marginal = utils.plot_v2_marginal_evolution(
                list(v_traj), list(t_traj),
                bounds_v=[(-0.8, 0.8)],
                bins_per_side=400,
                theta=theta,
                logy=False,
                title=run.name
            )
            wandb.log({"v2_marginal_evolution": wandb.Image(fig_v2_marginal)}, step=step)
            plt.close(fig_v2_marginal)
            print(f"  Saved: v2_marginal_evolution")

        print(f"  Successfully backfilled run: {run.name}")

    except Exception as e:
        print(f"  Error processing run: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "="*60)
print("Backfill complete!")
print("="*60)
