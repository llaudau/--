#!/usr/bin/env python3
"""
Wilson Loop Analysis
- Performs jackknife resampling on Wilson loop data
- Fits W(R,T) = A * exp(-V(R) * T) to extract potential V(R)
- Plots the Wilson loops and potential
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Configuration
T = 12          # Temporal extent
S = 6           # Spatial extent (x direction)
n_samples = 10000  # Number of configurations

# File paths
data_dir = "/home/khw/Documents/Git_repository/hmc_gpu/hmc_results/t12_s6_beta6.0"
output_dir = "/home/khw/Documents/Git_repository/qcd/data_analyze"

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


def fit_wilson_loops(data_jk):
    """
    Fit Wilson loops to extract potential V(R) for each spatial separation R.
    data_jk: [n_samples, T, S]
    Returns: V[R], A[R], and their errors
    """
    n_samples = data_jk.shape[0]
    n_r = data_jk.shape[2]  # S (spatial extent)
    
    V = np.zeros(n_r)
    V_err = np.zeros(n_r)
    A = np.zeros(n_r)
    A_err = np.zeros(n_r)
    
    # Time values (use t = 1 to T-1 to avoid boundary effects)
    t = np.arange(1, T)
    
    for r_idx in range(n_r):
        # Get all configurations' Wilson loops for this r at all times
        W_r = data_jk[:, 1:T, r_idx]  # [n_samples, T-1]
        
        # Average over configurations for fit
        W_mean = np.mean(W_r, axis=0)
        
        # Calculate covariance matrix for error estimation
        cov = np.cov(W_r.T) * (n_samples - 1)
        
        # Add small diagonal for numerical stability
        cov += np.eye(T-1) * 1e-8
        
        # Initial guess
        A0 = W_mean[0]
        V0 = 0.1
        
        try:
            # Fit using mean data
            popt, pcov = curve_fit(wilson_loop_exp, t, W_mean, p0=[A0, V0], sigma=cov, absolute_sigma=True)
            
            # Store amplitude and potential
            A[r_idx] = popt[0]
            V[r_idx] = popt[1]
            
            # Error from covariance
            V_err[r_idx] = np.sqrt(pcov[1, 1])
            A_err[r_idx] = np.sqrt(pcov[0, 0])
            
        except Exception as e:
            print(f"  Warning: Fit failed for r={r_idx+1}: {e}")
            V[r_idx] = np.nan
            V_err[r_idx] = np.nan
            A[r_idx] = np.nan
            A_err[r_idx] = np.nan
    
    return V, V_err, A, A_err


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
        if jk_idx % 1000 == 0:
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

# Spatial separations (in lattice units)
r = np.arange(1, S + 1)

print("\nResults:")
print("-" * 50)
print(f"{'R':>4} {'V(R)':>12} {'Error':>12} {'A(R)':>12} {'Error':>12}")
print("-" * 50)
for i in range(S):
    print(f"{r[i]:>4} {V_mean[i]:>12.6f} {V_err[i]:>12.6f} {A_mean[i]:>12.6f} {A_err[i]:>12.6f}")

# ============= PLOTTING =============

# Plot 1: Wilson loops W(R, T) vs T for different R
print("\nPlotting Wilson loops...")
fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
axes1 = axes1.flatten()

t_plot = np.arange(1, T)

for r_idx in range(min(S, 6)):
    ax = axes1[r_idx]
    
    # Get mean and error from jackknife
    W_mean_r = np.mean(data_jk[:, :, r_idx], axis=0)
    W_err_r = np.std(data_jk[:, :, r_idx], axis=0) * np.sqrt(n_samples - 1)
    
    # Plot data points
    ax.errorbar(t_plot, W_mean_r[1:], W_err_r[1:], fmt='o', capsize=3, label='Data')
    
    # Plot fit curve
    t_fit = np.linspace(1, T-1, 100)
    W_fit = A_mean[r_idx] * np.exp(-V_mean[r_idx] * t_fit)
    ax.plot(t_fit, W_fit, '-', label=f'Fit: V={V_mean[r_idx]:.4f}')
    
    ax.set_xlabel('t/a')
    ax.set_ylabel('W(R,t)')
    ax.set_title(f'R = {r_idx + 1}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Wilson Loops W(R,t) vs t (T={T}, S={S}, n={n_samples})', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'wilson_loop_vs_t.png'), dpi=300)
plt.close()
print(f"  Saved: wilson_loop_vs_t.png")

# Plot 2: Potential V(R) vs R
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.errorbar(r, V_mean, yerr=V_err, fmt='o', capsize=4, markersize=8, 
             color='blue', ecolor='blue', label='V(R) from Wilson loops')

ax2.set_xlabel('R/a (spatial separation)', fontsize=12)
ax2.set_ylabel('V(R) (potential)', fontsize=12)
ax2.set_title(f'Quark Potential V(R) from Wilson Loops (T={T}, S={S})', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Add annotation with fit parameters
textstr = '\n'.join([f'R={i+1}: V={V_mean[i]:.4f}±{V_err[i]:.4f}' for i in range(min(3, S))])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
          verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'potential_vs_R.png'), dpi=300)
plt.close()
print(f"  Saved: potential_vs_R.png")

# Plot 3: Log(W) vs t to check exponential behavior
fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))
axes3 = axes3.flatten()

for r_idx in range(min(S, 6)):
    ax = axes3[r_idx]
    
    # Get mean from jackknife
    W_mean_r = np.mean(data_jk[:, :, r_idx], axis=0)
    
    # Log of Wilson loop
    log_W = np.log(W_mean_r[1:] + 1e-10)  # avoid log(0)
    
    ax.plot(t_plot, log_W, 'o-', markersize=4)
    
    # Linear fit to log(W) = log(A) - V*t
    slope, intercept = np.polyfit(t_plot, log_W, 1)
    
    ax.plot(t_plot, intercept + slope * t_plot, '--', color='red', 
            label=f'Linear fit: V={-slope:.4f}')
    
    ax.set_xlabel('t/a')
    ax.set_ylabel('log(W(R,t))')
    ax.set_title(f'R = {r_idx + 1}')
    ax.legend()
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
                  label=f'R={r_idx+1}')

ax4a.set_xlabel('t/a', fontsize=12)
ax4a.set_ylabel('W(R,t)', fontsize=12)
ax4a.set_title('Wilson Loops', fontsize=14)
ax4a.legend()
ax4a.grid(True, alpha=0.3)

# Right: Potential
ax4b.errorbar(r, V_mean, yerr=V_err, fmt='o-', capsize=4, markersize=8, 
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

print("\n" + "="*50)
print("Analysis Complete!")
print(f"Output files saved to: {output_dir}")
print("="*50)
