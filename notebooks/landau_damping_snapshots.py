# %% [markdown]
# # Landau Damping: v1-v2 Heatmaps & v2 Marginals
# Download trajectory snapshots from 4 wandb runs and plot them.

# %%
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(".")), ""))
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import matplotlib.pyplot as plt
import wandb

from src import utils

# %%
# Run IDs and project
PROJECT = "vilin97-uw/vlasov_landau_damping"
RUN_IDS = ["p2ukk27a", "fvf9wspt", "8cn1v3tu", "l4mvl65b", "grwpikyp", "u1li0uuy"]

api = wandb.Api()

# Download all trajectory data
run_data = {}
for rid in RUN_IDS:
    run = api.run(f"{PROJECT}/{rid}")
    config = run.config
    label = run.name.replace("kde", "blob")
    print(f"Run: {label} (id={rid}), dv={config.get('dv')}, C={config.get('C')}, score_method={config.get('score_method')}, n={config.get('n')}")

    # Find and download trajectory artifact
    arts = list(run.logged_artifacts())
    traj_art = [a for a in arts if "snapshots" in a.name][0]
    art_dir = traj_art.download()

    npz_files = [f for f in os.listdir(art_dir) if f.endswith(".npz")]
    data = np.load(os.path.join(art_dir, npz_files[0]), allow_pickle=True)

    v_traj = [np.array(v, dtype=np.float64) for v in data["v_traj"]]
    t_traj = data["t_traj"]

    run_data[rid] = {
        "label": label,
        "v_traj": v_traj,
        "t_traj": list(t_traj),
        "config": config,
    }
    print(f"  Loaded {len(t_traj)} snapshots at t = {list(t_traj)}")

# %%
# Save all plots
SAVE_DIR = "/Users/vasil/Github/-ICLR-workshop-Vlasov-Landau-SBTM/wandb_downloads/damping-v1v2-phase-space"
os.makedirs(SAVE_DIR, exist_ok=True)

# v1-v2 Marginal Heatmaps
for rid, rd in run_data.items():
    fig = utils.plot_v1v2_marginal_snapshots(
        rd["v_traj"],
        rd["t_traj"],
        bounds_v=[(-4.0, 4.0), (-4.0, 4.0)],
        bins_per_side=200,
        title=rd["label"],
    )
    fig.savefig(os.path.join(SAVE_DIR, f"{rd['label']}_v1v2_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# %%
# v2 Marginal Evolution (log scale)
# For Landau damping with Maxwellian initial condition, theta = 1
theta = 1.0

for rid, rd in run_data.items():
    fig = utils.plot_v2_marginal_evolution(
        rd["v_traj"],
        rd["t_traj"],
        bounds_v=[(-4.0, 4.0)],
        bins_per_side=400,
        theta=theta,
        title=rd["label"],
        logy=True,
    )
    fig.savefig(os.path.join(SAVE_DIR, f"{rd['label']}_v2_marginal_log.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# %%
