#%%
# Load trajectories from wandb and plot L2 distance to steady state and evolution plots
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb

from src import utils

# Configuration
PROJECT_NAME = "weibel"
NUM_EXPERIMENTS = 20

# Initialize wandb API
api = wandb.Api()

# Get the last N runs from the project
runs = api.runs(f"{PROJECT_NAME}", order="-created_at")
runs_list = list(runs)[:NUM_EXPERIMENTS]

print(f"Found {len(runs_list)} runs")

for run in runs_list:
    print(f"\nProcessing run: {run.name} (id: {run.id})")

    # Get run config
    config = run.config
    dv = config.get("dv", 2)
    beta = config.get("beta", 1e-2)
    c = config.get("c", 0.3)

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

        x_traj = data['x_traj']
        v_traj = data['v_traj']
        t_traj = data['t_traj']

        print(f"  Loaded trajectories: {len(t_traj)} snapshots, t={t_traj}")

        # Create output directory
        outdir = f"plots/{run.name}"
        os.makedirs(outdir, exist_ok=True)

        # 1. Compute L2 distance to steady state for each snapshot
        l2_distances = []
        for i, (v_snap, t_snap) in enumerate(zip(v_traj, t_traj)):
            l2_dist = utils.compute_l2_distance_to_gaussian(v_snap, dv, theta=theta)
            l2_distances.append(l2_dist)
            print(f"    t={t_snap:.1f}: L2 distance = {l2_dist:.6f}")

        # Plot L2 distance vs time
        fig_l2, ax_l2 = plt.subplots(figsize=(8, 5))
        ax_l2.plot(t_traj, l2_distances, 'o-', linewidth=2, markersize=8)
        ax_l2.set_xlabel("Time")
        ax_l2.set_ylabel("L2 Distance to Steady State")
        ax_l2.set_title(f"L2 Distance to N(0, {theta:.4f})\n{run.name}")
        ax_l2.grid(True, alpha=0.3)
        ax_l2.set_yscale('log')
        fig_l2.tight_layout()
        fig_l2.savefig(os.path.join(outdir, "l2_distance.png"), dpi=150)
        plt.close(fig_l2)
        print(f"  Saved: {outdir}/l2_distance.png")

        # 2. Plot v2 at v1≈0 evolution
        fig_v2_zero = utils.plot_v2_at_v1_zero_evolution(
            list(v_traj), list(t_traj),
            bounds_v=[(-0.01, 0.01), (-3.0, 3.0)],
            bins_per_side=(1, 400),
            theta=theta
        )
        fig_v2_zero.suptitle(f"V2 Distribution at v1 ≈ 0\n{run.name}")
        fig_v2_zero.tight_layout()
        fig_v2_zero.savefig(os.path.join(outdir, "v2_at_v1_zero_evolution.png"), dpi=150)
        plt.close(fig_v2_zero)
        print(f"  Saved: {outdir}/v2_at_v1_zero_evolution.png")

        # 3. Plot v2 marginal evolution
        fig_v2_marginal = utils.plot_v2_marginal_evolution(
            list(v_traj), list(t_traj),
            bounds_v=[(-3.0, 3.0)],
            bins_per_side=400,
            theta=theta
        )
        fig_v2_marginal.suptitle(f"V2 Marginal Distribution\n{run.name}")
        fig_v2_marginal.tight_layout()
        fig_v2_marginal.savefig(os.path.join(outdir, "v2_marginal_evolution.png"), dpi=150)
        plt.close(fig_v2_marginal)
        print(f"  Saved: {outdir}/v2_marginal_evolution.png")

    except Exception as e:
        print(f"  Error processing run: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\nDone!")

# %%
# Load specific run and analyze variance
import os
import numpy as np
import wandb

api = wandb.Api()

# Specific run
# run_id = "fgtgnhyt" # blob
run_id = "5i4h538e" # sbtm
run = api.run(f"weibel/{run_id}")

config = run.config
dv = config.get("dv", 3)
beta = config.get("beta", 0.01)
c = config.get("c", 0.3)

if dv == 2:
    theta = (beta + c**2) / 2
else:  # dv == 3
    theta = beta / 2 + c**2 / 3

print(f"Run: {run.name}")
print(f"dv={dv}, beta={beta}, c={c}")
print(f"Theoretical steady-state variance theta = {theta:.6f}")

# Download artifact
artifacts = run.logged_artifacts()
traj_artifact = None
for artifact in artifacts:
    if "trajectories" in artifact.name:
        traj_artifact = artifact
        break

artifact_dir = traj_artifact.download()
npz_files = [f for f in os.listdir(artifact_dir) if f.endswith('.npz')]
npz_path = os.path.join(artifact_dir, npz_files[0])
data = np.load(npz_path)

x_traj = data['x_traj']
v_traj = data['v_traj']
t_traj = data['t_traj']

print(f"\nSnapshots at t = {t_traj}")
print(f"\n{'t':>8} | {'Var(v1)':>12} | {'Var(v2)':>12} | {'Var(v3)':>12} | {'Mean Var':>12} | {'theta':>12} | {'Diff':>12}")
print("-" * 90)

for t, v_snap in zip(t_traj, v_traj):
    var_v1 = np.var(v_snap[:, 0])
    var_v2 = np.var(v_snap[:, 1])
    var_v3 = np.var(v_snap[:, 2]) if dv == 3 else None

    if dv == 3:
        mean_var = (var_v1 + var_v2 + var_v3) / 3
        print(f"{t:8.1f} | {var_v1:12.6f} | {var_v2:12.6f} | {var_v3:12.6f} | {mean_var:12.6f} | {theta:12.6f} | {mean_var - theta:+12.6f}")
    else:
        mean_var = (var_v1 + var_v2) / 2
        print(f"{t:8.1f} | {var_v1:12.6f} | {var_v2:12.6f} | {'N/A':>12} | {mean_var:12.6f} | {theta:12.6f} | {mean_var - theta:+12.6f}")

#%%
# plot the v1=0 slice of v1-v2 marginal, and also the v1-v2 marginal itself.
import matplotlib.pyplot as plt
from src import utils
# Reimport utils to ensure latest version is loaded
import importlib
importlib.reload(utils)

lim = 0.8

# 1. Plot v1-v2 marginal distribution
fig_v1v2 = utils.plot_v1v2_marginal_snapshots(
    list(v_traj), list(t_traj),
    bounds_v=[(-lim, lim), (-lim, lim)],
    bins_per_side=200,
    title=run.name
)
plt.show()

# 2. Plot v2 at v1≈0 slice
fig_v2_slice = utils.plot_v2_at_v1_zero_evolution(
    list(v_traj), list(t_traj),
    bounds_v=[(-0.05, 0.05), (-lim, lim)],
    bins_per_side=(1, 200),
    theta=theta,
    logy=True,
    title=run.name
)
plt.show()

# 3. Plot v2 marginal distribution
fig_v2_marginal = utils.plot_v2_marginal_evolution(
    list(v_traj), list(t_traj),
    bounds_v=[(-lim, lim)],
    bins_per_side=200,
    theta=theta,
    title=run.name,
    logy=True
)
plt.show()

# 4. Compute L2 distance of v2 marginal to Gaussian
print(f"L2 distance of v2 marginal to N(0, {theta:.6f}):")
for v_snap, t_snap in zip(v_traj, t_traj):
    l2_dist = utils.compute_l2_distance_to_gaussian(v_snap[:, 1:2], dv=1, theta=theta, bounds_v=[(-lim, lim)], bins=200)
    print(f"  t={t_snap:.1f}: L2 = {l2_dist:.6f}")

# %%
# Debug: Compare actual distribution to Gaussian with variance theta
import matplotlib.pyplot as plt

# Take the last snapshot (t=125)
v_final = v_traj[-1]
t_final = t_traj[-1]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, (ax, label) in enumerate(zip(axes, ['v1', 'v2', 'v3'])):
    if i >= dv:
        ax.axis('off')
        continue

    v_component = v_final[:, i]

    # Histogram of actual data
    counts, bins, _ = ax.hist(v_component, bins=100, density=True, alpha=0.7, label=f'Data (var={np.var(v_component):.6f})')

    # Gaussian with variance theta
    v_grid = np.linspace(bins[0], bins[-1], 200)
    gaussian = np.exp(-0.5 * v_grid**2 / theta) / np.sqrt(2 * np.pi * theta)
    ax.plot(v_grid, gaussian, 'r-', linewidth=2, label=f'N(0, {theta:.4f})')

    # Also plot Gaussian with actual variance for comparison
    actual_var = np.var(v_component)
    gaussian_actual = np.exp(-0.5 * v_grid**2 / actual_var) / np.sqrt(2 * np.pi * actual_var)
    ax.plot(v_grid, gaussian_actual, 'g--', linewidth=2, label=f'N(0, {actual_var:.4f})')

    ax.set_xlabel(label)
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_title(f'{label} at t={t_final}')

plt.suptitle(f'Distribution comparison (theta={theta:.6f})')
plt.tight_layout()
plt.show()

# %%
# Check if the issue is in compute_l2_distance_to_gaussian
# Let's manually compute what's happening

from src.utils import density_on_regular_grid
import jax.numpy as jnp

# Use same parameters as compute_l2_distance_to_gaussian
bins = 200
bounds_v = [(-3.0, 3.0)] * dv

# Get density on regular grid
density = density_on_regular_grid(
    v_final, bounds_v=bounds_v, bins_per_side=bins, smooth_sigma_bins=0.0
)

print(f"Density grid shape: {density.shape}")
print(f"Density sum * dx^dv: {float(jnp.sum(density)) * (6.0/bins)**dv:.6f} (should be ~1)")
print(f"Density min: {float(jnp.min(density)):.6f}, max: {float(jnp.max(density)):.6f}")

# Create coordinate grids
bounds_v_arr = jnp.asarray(bounds_v)
lo, hi = bounds_v_arr[:, 0], bounds_v_arr[:, 1]
dx = (hi - lo) / bins

axes_grid = [lo[d] + (jnp.arange(bins) + 0.5) * dx[d] for d in range(dv)]
mesh = jnp.meshgrid(*axes_grid, indexing="ij")

# Compute Gaussian with variance theta
v_squared = sum(m**2 for m in mesh)
gaussian = jnp.exp(-0.5 * v_squared / theta) / ((2.0 * jnp.pi * theta) ** (dv / 2.0))

print(f"\nGaussian sum * dx^dv: {float(jnp.sum(gaussian)) * (6.0/bins)**dv:.6f} (should be ~1)")
print(f"Gaussian min: {float(jnp.min(gaussian)):.6f}, max: {float(jnp.max(gaussian)):.6f}")

# The issue: bounds [-3, 3] might not capture enough of the distribution
# Check what fraction of particles are within bounds
in_bounds = np.all((v_final >= -3.0) & (v_final <= 3.0), axis=1)
print(f"\nFraction of particles within [-3, 3]^{dv}: {np.mean(in_bounds):.6f}")

# What are the actual data ranges?
print(f"\nActual data ranges:")
for i in range(dv):
    print(f"  v{i+1}: [{v_final[:, i].min():.4f}, {v_final[:, i].max():.4f}]")

# Check: what's the expected range for N(0, theta)?
# 99.7% of data within 3*sigma = 3*sqrt(theta)
sigma = np.sqrt(theta)
print(f"\nFor N(0, {theta:.4f}): sigma = {sigma:.4f}")
print(f"  3-sigma range: [{-3*sigma:.4f}, {3*sigma:.4f}]")
print(f"  Current bounds: [-3, 3]")
print(f"  Bounds in sigma units: [{-3/sigma:.1f} sigma, {3/sigma:.1f} sigma]")

# %%
# The data ranges to ~4.5 sigma but Gaussian 3-sigma is only 0.56
# This means the distribution has HEAVIER TAILS than Gaussian
# Let's look at 1D marginals more carefully

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, (ax, label) in enumerate(zip(axes, ['v1', 'v2', 'v3'])):
    if i >= dv:
        ax.axis('off')
        continue

    v_component = v_final[:, i]

    # Use tighter bounds for histogram - around the data
    hist_bounds = (-1.0, 1.0)
    counts, bin_edges, _ = ax.hist(v_component, bins=100, range=hist_bounds, density=True, alpha=0.7, label='Data')

    # Gaussian with variance theta
    v_grid = np.linspace(hist_bounds[0], hist_bounds[1], 200)
    gaussian = np.exp(-0.5 * v_grid**2 / theta) / np.sqrt(2 * np.pi * theta)
    ax.plot(v_grid, gaussian, 'r-', linewidth=2, label=f'N(0, {theta:.4f})')

    ax.set_xlabel(label)
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_title(f'{label} at t={t_final}')
    ax.set_xlim(hist_bounds)

plt.suptitle(f'Distribution comparison - tighter view')
plt.tight_layout()
plt.show()

# %%
# Check the tails - what fraction of particles are beyond 3*sigma?
print("Tail analysis:")
for sigma_mult in [3, 4, 5]:
    threshold = sigma_mult * sigma
    for i, label in enumerate(['v1', 'v2', 'v3'][:dv]):
        frac_beyond = np.mean(np.abs(v_final[:, i]) > threshold)
        # Expected for Gaussian
        from scipy import stats
        expected_frac = 2 * (1 - stats.norm.cdf(sigma_mult))
        print(f"  |{label}| > {sigma_mult}*sigma ({threshold:.4f}): {frac_beyond*100:.4f}% (Gaussian expects {expected_frac*100:.4f}%)")
    print()

# %%
# Let's compute L2 with appropriate bounds (e.g., 5*sigma)
# and see how the distance changes

from src.utils import compute_l2_distance_to_gaussian

# With default bounds [-3, 3]
l2_default = compute_l2_distance_to_gaussian(v_final, dv, theta=theta, bounds_v=[(-3.0, 3.0)] * dv)
print(f"L2 with bounds [-3, 3]: {l2_default:.6f}")

# With tighter bounds around the data (e.g., 5*sigma ~ 0.94)
bound_5sigma = 5 * sigma
l2_5sigma = compute_l2_distance_to_gaussian(v_final, dv, theta=theta, bounds_v=[(-bound_5sigma, bound_5sigma)] * dv)
print(f"L2 with bounds [-5sigma, 5sigma] = [{-bound_5sigma:.3f}, {bound_5sigma:.3f}]: {l2_5sigma:.6f}")

# With bounds matching actual data range
bound_data = 1.0
l2_data = compute_l2_distance_to_gaussian(v_final, dv, theta=theta, bounds_v=[(-bound_data, bound_data)] * dv)
print(f"L2 with bounds [-1, 1]: {l2_data:.6f}")

# %%
# The 1D marginals are perfect but 3D L2 is large
# Possible issues:
# 1. Correlations between velocity components
# 2. 3D density estimation issues (sparse bins)

# Check correlations
print("Correlation matrix:")
corr = np.corrcoef(v_final.T)
print(corr)

print("\nCovariance matrix:")
cov = np.cov(v_final.T)
print(cov)
print(f"\nExpected covariance (isotropic): {theta:.6f} * I")

# %%
# Check 2D marginals - are v1-v2, v1-v3, v2-v3 independent?
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

pairs = [(0, 1, 'v1', 'v2'), (0, 2, 'v1', 'v3'), (1, 2, 'v2', 'v3')]
for ax, (i, j, li, lj) in zip(axes, pairs):
    if j >= dv:
        ax.axis('off')
        continue

    ax.hist2d(v_final[:, i], v_final[:, j], bins=50, range=[[-0.8, 0.8], [-0.8, 0.8]], density=True, cmap='viridis')
    ax.set_xlabel(li)
    ax.set_ylabel(lj)
    ax.set_aspect('equal')
    ax.set_title(f'{li}-{lj} (corr={corr[i,j]:.4f})')

plt.suptitle('2D marginal distributions')
plt.tight_layout()
plt.show()

# %%
# The issue might be with sparse 3D binning - 200^3 = 8 million bins for 1 million particles
# Many bins are empty, causing numerical issues

# Let's compute 1D L2 distances instead
print("1D L2 distances (per component):")
for i, label in enumerate(['v1', 'v2', 'v3'][:dv]):
    l2_1d = compute_l2_distance_to_gaussian(v_final[:, i:i+1], 1, theta=theta, bounds_v=[(-1.0, 1.0)])
    print(f"  {label}: {l2_1d:.6f}")

# Product of 1D Gaussians should equal 3D Gaussian for independent variables
# So if marginals are good but joint is bad, either:
# 1. Correlations exist
# 2. Density estimation in 3D is unreliable

# %%
# Let's try computing L2 with fewer bins to reduce sparsity
print("\n3D L2 with different bin counts (bounds [-1, 1]):")
for nbins in [20, 30, 40, 50, 75, 100, 150, 200]:
    l2_val = compute_l2_distance_to_gaussian(v_final, dv, theta=theta, bounds_v=[(-1.0, 1.0)] * dv, bins=nbins)
    n_bins_total = nbins ** dv
    particles_per_bin = len(v_final) / n_bins_total
    print(f"  bins={nbins:3d} ({n_bins_total:>10.0e} total, ~{particles_per_bin:>6.1f} particles/bin): L2 = {l2_val:.6f}")

# %%
# Sanity check: compute L2 distance for a KNOWN Gaussian sample
# This tells us the "noise floor" of the L2 metric due to finite sampling

n_particles = len(v_final)
print(f"Sanity check: L2 distance for {n_particles} samples from true N(0, {theta:.4f})")

# Generate samples from the true distribution
np.random.seed(42)
v_true_gaussian = np.random.randn(n_particles, dv) * np.sqrt(theta)

print("\n3D L2 for TRUE Gaussian samples (bounds [-1, 1]):")
for nbins in [20, 30, 40, 50, 75, 100]:
    l2_true = compute_l2_distance_to_gaussian(v_true_gaussian, dv, theta=theta, bounds_v=[(-1.0, 1.0)] * dv, bins=nbins)
    l2_data = compute_l2_distance_to_gaussian(v_final, dv, theta=theta, bounds_v=[(-1.0, 1.0)] * dv, bins=nbins)
    n_bins_total = nbins ** dv
    particles_per_bin = n_particles / n_bins_total
    print(f"  bins={nbins:3d} (~{particles_per_bin:>6.1f} p/bin): TRUE Gaussian L2 = {l2_true:.6f}, Data L2 = {l2_data:.6f}, Ratio = {l2_data/l2_true:.2f}")

# %%
# If the ratio is close to 1, then our data is essentially indistinguishable from a true Gaussian
# The L2 "distance" is just sampling noise

# Let's also check 1D
print("\n1D L2 for TRUE Gaussian vs Data:")
for i, label in enumerate(['v1', 'v2', 'v3'][:dv]):
    l2_true_1d = compute_l2_distance_to_gaussian(v_true_gaussian[:, i:i+1], 1, theta=theta, bounds_v=[(-1.0, 1.0)])
    l2_data_1d = compute_l2_distance_to_gaussian(v_final[:, i:i+1], 1, theta=theta, bounds_v=[(-1.0, 1.0)])
    print(f"  {label}: TRUE L2 = {l2_true_1d:.6f}, Data L2 = {l2_data_1d:.6f}, Ratio = {l2_data_1d/l2_true_1d:.2f}")

# %%
# CONCLUSION: The L2 distance is dominated by sampling noise, not actual deviation from Gaussian
# The ratio ~1.0 means our data IS the target Gaussian (within statistical noise)

# Better metrics for convergence to Gaussian:
# 1. Variance error: |Var(v) - theta| / theta  (relative error in variance)
# 2. Covariance error: ||Cov(v) - theta*I||_F / (theta * sqrt(dv))  (Frobenius norm)
# 3. Kurtosis: should be 3 for Gaussian (excess kurtosis = 0)
# 4. Skewness: should be 0 for Gaussian

from scipy.stats import kurtosis, skew
import importlib
from src import utils

print("=" * 60)
print("BETTER CONVERGENCE METRICS")
print("=" * 60)

# Variance error
var_error = np.abs(np.mean([np.var(v_final[:, i]) for i in range(dv)]) - theta) / theta
print(f"\n1. Relative variance error: {var_error:.6f} (should be ~0)")

# Covariance error (off-diagonal should be 0, diagonal should be theta)
cov_target = theta * np.eye(dv)
cov_error = np.linalg.norm(cov - cov_target, 'fro') / (theta * np.sqrt(dv))
print(f"2. Relative covariance error (Frobenius): {cov_error:.6f} (should be ~0)")

# Kurtosis (excess kurtosis should be 0 for Gaussian)
print(f"3. Excess kurtosis (should be ~0 for Gaussian):")
for i, label in enumerate(['v1', 'v2', 'v3'][:dv]):
    kurt = kurtosis(v_final[:, i])  # scipy returns excess kurtosis
    print(f"   {label}: {kurt:.6f}")

# Skewness (should be 0 for Gaussian)
print(f"4. Skewness (should be ~0 for Gaussian):")
for i, label in enumerate(['v1', 'v2', 'v3'][:dv]):
    sk = skew(v_final[:, i])
    print(f"   {label}: {sk:.6f}")

# Compare with true Gaussian samples
print(f"\nFor comparison, TRUE Gaussian samples:")
print(f"   Excess kurtosis: {[f'{kurtosis(v_true_gaussian[:, i]):.6f}' for i in range(dv)]}")
print(f"   Skewness: {[f'{skew(v_true_gaussian[:, i]):.6f}' for i in range(dv)]}")

# %%
# Let's compute these metrics for ALL snapshots to see convergence over time
print("\nConvergence over time:")
print(f"{'t':>8} | {'Var Error':>12} | {'Cov Error':>12} | {'Mean Kurt':>12} | {'Mean Skew':>12}")
print("-" * 70)

for t, v_snap in zip(t_traj, v_traj):
    # Variance error
    mean_var = np.mean([np.var(v_snap[:, i]) for i in range(dv)])
    var_err = np.abs(mean_var - theta) / theta

    # Covariance error
    cov_snap = np.cov(v_snap.T)
    cov_target = theta * np.eye(dv)
    cov_err = np.linalg.norm(cov_snap - cov_target, 'fro') / (theta * np.sqrt(dv))

    # Mean kurtosis
    mean_kurt = np.mean([kurtosis(v_snap[:, i]) for i in range(dv)])

    # Mean skewness
    mean_skew = np.mean([np.abs(skew(v_snap[:, i])) for i in range(dv)])

    print(f"{t:8.1f} | {var_err:12.6f} | {cov_err:12.6f} | {mean_kurt:12.6f} | {mean_skew:12.6f}")

# %%
