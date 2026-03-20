#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

T = 24
S = 12
beta = 5.7
n_samples = 10000
plot_steps = [0, 10, 20, 29]

base_path = "/home/khw/Documents/Git_repository/hmc_gpu/hmc_results"
output_dir = f"/home/khw/Documents/Git_repository/qcd/data_analyze/result4/T{T}_S{S}_beta{beta:.1f}"
os.makedirs(output_dir, exist_ok=True)

folder_name = f"t{T}_s{S}_beta{beta:.1f}"
data_path = os.path.join(base_path, folder_name)

topo_file = os.path.join(data_path, "topo_charge_data.txt")
topo_data = np.loadtxt(topo_file)
smear_steps = len(topo_data) // n_samples
print(f"Detected smear_steps: {smear_steps}")

plot_steps = [s for s in plot_steps if s < smear_steps]
if len(plot_steps) == 0:
    plot_steps = [0]

n_plots = len(plot_steps)
n_cols = 2
n_rows = (n_plots + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
if n_plots == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for idx, s in enumerate(plot_steps):
    ax = axes[idx]
    start = s * n_samples
    end = (s + 1) * n_samples
    topo_s = topo_data[start:end]
    
    bins = np.linspace(min(topo_s), max(topo_s), 30)
    
    ax.hist(topo_s, bins=bins, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Topological Charge', fontsize=11)
    ax.set_ylabel('Number of Configs', fontsize=11)
    ax.set_title(f'Smear step {s}', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    mean_val = np.mean(topo_s)
    std_val = np.std(topo_s)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.4f}')
    ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1, label=f'±σ: {std_val:.4f}')
    ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1)
    ax.legend(fontsize=9)

plt.suptitle(f'Topological Charge Distribution (T={T}, S={S}, β={beta:.1f})', fontsize=14)
plt.tight_layout()

output_file = os.path.join(output_dir, f"topo_hist_steps_{'_'.join(map(str, plot_steps))}.png")
plt.savefig(output_file, dpi=300)
plt.close()
print(f"Saved: {output_file}")
