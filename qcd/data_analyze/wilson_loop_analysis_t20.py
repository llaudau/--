#!/usr/bin/env python3
"""
Wilson Loop Analysis for T=20, S=10
- Performs jackknife resampling on Wilson loop data
- Fits W(R,T) = A * exp(-V(R) * T) to extract potential V(R)
- Plots the Wilson loops and potential with error bars
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Configuration
T = 20          # Temporal extent
S = 10          # Spatial extent (x direction)
n_samples = 10000  # Number of configurations

# File paths
data_dir = "/home/khw/Documents/Git_repository/qcd/data_analyze/result3/t20_s10_beta_6.0"
output_dir = "/home/khw/Documents/Git_repository/qcd/data_analyze/result3/t20_s10_beta_6.0"

# Load Wilson loop data
print("Loading Wilson loop data...")
wilson_file = os.path.join(data_dir, "wilson_loop_data.txt")
data = np.loadtxt(wilson_file)

print(f"Loaded {len(data)} data points")

# Reshape data: [t_ext, x_ext, trajectory]
# Format: for each t_ext, all x_ext values for all trajectories
n_wilson_loops = T * S
data = data.reshape(T, S, n_samples)

print(f"Data shape: {data.shape} (t, x, configs)")

# Transpose to: [configs, t, x] for easier handling
data = np.transpose(data, (2, 0, 1))
print(f"Reshaped to: {data.shape} (configs, t, x)")


def jackknife(data):
    """
    Perform jackknife resampling on the data.
    Input: data[configs, ...]
    Output: jackknife samples[configs, ...]
    """
    configs = data.shape[0]
    # Calculate the mean for each t, x
    mean = np.mean(data, axis=0)
    # Tile the mean to match configs
    mean_tiled = np.tile(mean, (configs, 1, 1))
    # Jackknife samples: exclude each config one at a time
    jackknife_samples = (mean_tiled * configs - data) / (configs - 1)
    return jackknife_samples


def wilson_loop_exp(t, A, V):
    """
    Wilson loop fitting function: W(T) = A * exp(-V * T)
    """
    return A * np.exp(-V * t)


def fit_wilson_loops_full_jackknife(data_jk):
    """
    Perform full jackknife analysis: fit each jackknife sample individually.
    This gives proper error estimation.
    """
    n_samples = data_jk.shape[0]
    n_r = data_jk.shape[2]  # S
    
    # Storage for jackknife fits
    V_jk = np.zeros((n_samples, n_r))
    A_jk = np.zeros((n_samples, n_r))
    
    t = np.arange(1, T)
    
    print("Fitting jackknife samples...")
    for jk_idx in range(n_samples):
        if jk_idx % 200 == 0:
            print(f"  Progress: {jk_idx}/{n_samples}")
        
        for r_idx in range(n_r):
            W_r = data_jk[jk_idx, 1:T, r_idx]
            
            try:
                A0 = W_r[0] if W_r[0] > 0 else 1.0
                V0 = 0.1
                popt, _ = curve_fit(wilson_loop_exp, t, W_r, p0=[A0, V0], maxfev=5000)
                A_jk[jk_idx, r_idx] = popt[0]
                V_jk[jk_idx, r_idx] = popt[1]
            except:
                A_jk[jk_idx, r_idx] = np.nan
                V_jk[jk_idx, r_idx] = np.nan
    
    # Calculate mean and error for V and A
    V_mean = np.nanmean(V_jk, axis=0)
    V_err = np.nanstd(V_jk, axis=0) * np.sqrt(n_samples - 1)
    
    A_mean = np.nanmean(A_jk, axis=0)
    A_err = np.nanstd(A_jk, axis=0) * np.sqrt(n_samples - 1)
    
    return V_mean, V_err, A_mean, A_err, V_jk, A_jk


# Perform jackknife resampling
print("\nPerforming jackknife resampling...")
data_jk = jackknife(data)
print(f"Jackknife samples shape: {data_jk.shape}")

# Fit Wilson loops using jackknife samples
print("\nFitting Wilson loops to extract potential...")
V_mean, V_err, A_mean, A_err, V_jk, A_jk = fit_wilson_loops_full_jackknife(data_jk)

# Spatial separations (in lattice units) - exclude R=S due to periodic boundary
r = np.arange(1, S + 1)
r_fit = np.arange(1, S)  # Exclude R=S (boundary effects)

print("\nResults:")
print("-" * 60)
print(f"{'R':>4} {'V(R)':>12} {'Error':>12} {'A(R)':>12} {'Error':>12}")
print("-" * 60)
for i in range(S):
    print(f"{r[i]:>4} {V_mean[i]:>12.6f} {V_err[i]:>12.6f} {A_mean[i]:>12.6f} {A_err[i]:>12.6f}")

# ============= PLOTTING =============

# Plot 1: Wilson loops W(R, T) vs T for different R
print("\nPlotting Wilson loops...")
fig1, axes1 = plt.subplots(2, 4, figsize=(20, 10))
axes1 = axes1.flatten()

t_plot = np.arange(1, T)

for r_idx in range(min(S, 8)):
    ax = axes1[r_idx]
    
    # Get mean and error from jackknife
    W_mean_r = np.mean(data_jk[:, :, r_idx], axis=0)
    W_err_r = np.std(data_jk[:, :, r_idx], axis=0) * np.sqrt(n_samples - 1)
    
    # Plot data points with error bars
    ax.errorbar(t_plot, W_mean_r[1:], W_err_r[1:], fmt='o', capsize=2, 
                 markersize=4, label='Data', color='blue')
    
    # Plot fit curve
    t_fit = np.linspace(1, T-1, 100)
    W_fit = A_mean[r_idx] * np.exp(-V_mean[r_idx] * t_fit)
    ax.plot(t_fit, W_fit, '-', color='red', linewidth=1.5,
            label=f'Fit: V={V_mean[r_idx]:.4f}')
    
    ax.set_xlabel('t/a', fontsize=10)
    ax.set_ylabel('W(R,t)', fontsize=10)
    ax.set_title(f'R = {r_idx + 1}', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Wilson Loops W(R,t) vs t (T={T}, S={S}, n={n_samples})', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'wilson_loop_vs_t.png'), dpi=300)
plt.close()
print(f"  Saved: wilson_loop_vs_t.png")

# Plot 2: Potential V(R) vs R with error bars
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Exclude R=S (boundary effects)
r_plot = r[:-1]
V_plot = V_mean[:-1]
V_plot_err = V_err[:-1]

ax2.errorbar(r_plot, V_plot, yerr=V_plot_err, fmt='o', capsize=4, markersize=8, 
             color='blue', ecolor='blue', label='V(R) from Wilson loops')

ax2.set_xlabel('R/a (spatial separation)', fontsize=12)
ax2.set_ylabel('V(R) (potential)', fontsize=12)
ax2.set_title(f'Quark Potential V(R) from Wilson Loops (T={T}, S={S})', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# Add annotation with fit parameters
textstr = '\n'.join([f'R={i+1}: V={V_mean[i]:.4f}±{V_err[i]:.4f}' for i in range(min(5, S-1))])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
          verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'potential_vs_R.png'), dpi=300)
plt.close()
print(f"  Saved: potential_vs_R.png")

# Plot 3: Log(W) vs t to check exponential behavior
fig3, axes3 = plt.subplots(2, 4, figsize=(20, 10))
axes3 = axes3.flatten()

for r_idx in range(min(S, 8)):
    ax = axes3[r_idx]
    
    # Get mean from jackknife
    W_mean_r = np.mean(data_jk[:, :, r_idx], axis=0)
    
    # Log of Wilson loop
    log_W = np.log(np.abs(W_mean_r[1:]) + 1e-10)  # avoid log(0)
    
    ax.plot(t_plot, log_W, 'o-', markersize=4, color='blue')
    
    # Linear fit to log(W) = log(A) - V*t
    slope, intercept = np.polyfit(t_plot, log_W, 1)
    
    ax.plot(t_plot, intercept + slope * t_plot, '--', color='red', linewidth=1.5,
            label=f'V={-slope:.4f}')
    
    ax.set_xlabel('t/a', fontsize=10)
    ax.set_ylabel('log|W(R,t)|', fontsize=10)
    ax.set_title(f'R = {r_idx + 1}', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Log(Wilson Loop) vs t (Linear Fit gives -V)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'log_wilson_loop.png'), dpi=300)
plt.close()
print(f"  Saved: log_wilson_loop.png")

# Plot 4: Combined summary plot
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Wilson loops
for r_idx in range(0, S, 2):  # Every other R for clarity
    W_mean_r = np.mean(data_jk[:, :, r_idx], axis=0)
    W_err_r = np.std(data_jk[:, :, r_idx], axis=0) * np.sqrt(n_samples - 1)
    ax4a.errorbar(t_plot, W_mean_r[1:], W_err_r[1:], fmt='o-', capsize=2, 
                  markersize=3, label=f'R={r_idx+1}')

ax4a.set_xlabel('t/a', fontsize=12)
ax4a.set_ylabel('W(R,t)', fontsize=12)
ax4a.set_title('Wilson Loops', fontsize=14)
ax4a.legend(fontsize=8, ncol=2)
ax4a.grid(True, alpha=0.3)

# Right: Potential with error bars
ax4b.errorbar(r_plot, V_plot, yerr=V_plot_err, fmt='o-', capsize=4, markersize=8, 
              color='darkgreen', ecolor='darkgreen')
ax4b.set_xlabel('R/a', fontsize=12)
ax4b.set_ylabel('V(R)', fontsize=12)
ax4b.set_title('Static Quark Potential', fontsize=14)
ax4b.grid(True, alpha=0.3)

plt.suptitle(f'Wilson Loop Analysis: T={T}, S={S}, n={n_samples}', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'wilson_loop_summary.png'), dpi=300)
plt.close()
print(f"  Saved: wilson_loop_summary.png")

# Save numerical results to file
results_file = os.path.join(output_dir, 'potential_results.txt')
with open(results_file, 'w') as f:
    f.write("# Wilson Loop Analysis Results\n")
    f.write(f"# T={T}, S={S}, n_samples={n_samples}\n")
    f.write("#\n")
    f.write("# R\tV(R)\tV_err\tA(R)\tA_err\n")
    for i in range(S):
        f.write(f"{r[i]}\t{V_mean[i]:.6f}\t{V_err[i]:.6f}\t{A_mean[i]:.6f}\t{A_err[i]:.6f}\n")
print(f"  Saved: potential_results.txt")

print("\n" + "="*50)
print("Analysis Complete!")
print(f"Output files saved to: {output_dir}")
print("="*50)
