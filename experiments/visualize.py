#%%
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = "/mmfs1/gscratch/amath/vilin/Vlasov-Landau-SBTM"
example_name = "two_stream"
statistics_root = os.path.join(ROOT, "data", "statistics")
folders = [f for f in os.listdir(statistics_root) if f.startswith(example_name)]
folders.sort()

fig, axs = plt.subplots(2, 2, figsize=(20, 10))
axs = axs.flatten()

for folder in folders:
    folder_path = os.path.join(statistics_root, folder)
    try:
        EK = np.load(os.path.join(folder_path, 'kinetic_energy.npy'))
        EE = np.load(os.path.join(folder_path, 'electric_energy.npy'))
        E_L2 = np.load(os.path.join(folder_path, 'electric_field_norm.npy'))
    except FileNotFoundError:
        print(f"Missing data in {folder_path}, skipping.")
        continue

    steps = np.arange(len(EK))
    if "N1000000" in folder:
        axs[0].plot(steps, EK + EE, label=folder)
        axs[1].plot(steps, EK, label=folder)
        axs[2].plot(steps, EE, label=folder)
        axs[3].plot(steps, E_L2, label=folder)

axs[0].set_title("Total Energy")
axs[1].set_title("Kinetic Energy")
axs[2].set_title("Electric Energy")
axs[3].set_title("Electric Field L2 Norm")

for ax in axs:
    ax.set_xlabel("Time Step")
    ax.legend()
    ax.grid(True)

axs[0].set_ylabel("Energy")
axs[1].set_ylabel("Energy")
axs[2].set_ylabel("Energy")
axs[3].set_ylabel("Norm")

plt.suptitle(f"Energy Statistics for {example_name}")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()