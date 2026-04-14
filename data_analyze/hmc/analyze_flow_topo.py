#!/usr/bin/env python3
"""
analyze_flow_topo.py
--------------------
Comprehensive HMC analysis: plaquette history, topological charge,
gradient flow evolution, energy density, autocorrelation.

Data files (new format):
  Plaq.csv   -- traj, plaq
  smear.csv  -- traj, t_flow, Q, E

Auto-discovers all lattice datasets under results/.
Outputs plots to plots/{dataset}/ subdirectories.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import sys
import glob

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "../../hmc/results")
PLOT_ROOT    = os.path.join(os.path.dirname(__file__), "plots")
SQRT8T0_PHYS = 0.415  # fm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def jackknife_samples_2d(data):
    n = data.shape[0]
    total = np.sum(data, axis=0)
    return (total[np.newaxis, :] - data) / (n - 1)

def jackknife_stats_2d(jk):
    n = jk.shape[0]
    mean = np.mean(jk, axis=0)
    err = np.sqrt((n - 1) * np.var(jk, axis=0))
    return mean, err

def jackknife_stats(jk):
    n = len(jk)
    mean = np.mean(jk)
    err = np.sqrt((n - 1) * np.var(jk))
    return mean, err

def autocorrelation(data, max_lag=None):
    n = len(data)
    if max_lag is None:
        max_lag = min(n // 2, 500)
    mean = np.mean(data)
    var = np.var(data)
    if var == 0:
        return np.zeros(max_lag)
    rho = np.zeros(max_lag)
    for t in range(max_lag):
        rho[t] = np.mean((data[:n-t] - mean) * (data[t:] - mean)) / var
    return rho

def find_t0_interp(ft, t2E_vals, target=0.3):
    for i in range(len(t2E_vals) - 1):
        if (t2E_vals[i] - target) * (t2E_vals[i+1] - target) < 0:
            t_lo, t_hi = ft[i], ft[i+1]
            v_lo, v_hi = t2E_vals[i], t2E_vals[i+1]
            return t_lo + (target - v_lo) * (t_hi - t_lo) / (v_hi - v_lo)
    return np.nan

def find_datasets(root):
    datasets = []
    for plaq in sorted(glob.glob(os.path.join(root, "*/*/flow/Plaq.csv"))):
        data_dir = os.path.dirname(plaq)
        smear = os.path.join(data_dir, "smear.csv")
        if os.path.isfile(smear):
            parts = data_dir.replace(root, "").strip("/").split("/")
            label = f"{parts[0]}/{parts[1]}"
            datasets.append((label, data_dir))
    return datasets

# ---------------------------------------------------------------------------
# Analyze one dataset
# ---------------------------------------------------------------------------
def analyze_one(label, data_dir, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    # Load Plaq.csv
    plaq_df = pd.read_csv(os.path.join(data_dir, "Plaq.csv"))
    n_traj = len(plaq_df)
    plaq_vals = plaq_df["plaq"].values
    traj_ids = plaq_df["traj"].values

    # Load smear.csv
    smear_df = pd.read_csv(os.path.join(data_dir, "smear.csv"))
    flow_times = np.sort(smear_df["t_flow"].unique())
    n_flow = len(flow_times)

    # Pivot to 2D arrays [n_traj, n_flow]
    Q_flow = smear_df.pivot(index="traj", columns="t_flow", values="Q").values
    E_flow = smear_df.pivot(index="traj", columns="t_flow", values="E").values

    print(f"\n[{label}] {n_traj} trajectories, {n_flow} flow steps, "
          f"t_flow: [{flow_times[0]:.3f}, {flow_times[-1]:.3f}]")
    print(f"  <Plaq> = {plaq_vals.mean():.6f} +/- {plaq_vals.std():.6f}")

    # Q at max flow time (for history, histogram, autocorrelation)
    Q_final = Q_flow[:, -1]

    # ---- Plot 1: Plaquette history ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(traj_ids, plaq_vals, '-', lw=0.5, color='steelblue')
    ax.set_xlabel('Trajectory')
    ax.set_ylabel('Plaquette')
    ax.set_title(f'Plaquette History [{label}]')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(plot_dir, 'plaquette_history.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"  Saved {out}")

    # ---- Plot 2: Q history & histogram ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(traj_ids, Q_final, '-', lw=0.5, color='darkorange')
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Trajectory')
    ax.set_ylabel(f'Q (at t = {flow_times[-1]:.2f})')
    ax.set_title('Topological Charge History')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(Q_final, bins=40, color='darkorange', alpha=0.7, edgecolor='k', lw=0.5)
    ax.axvline(np.mean(Q_final), color='red', ls='--', lw=1.5,
               label=f'<Q> = {np.mean(Q_final):.2f}')
    ax.axvline(np.mean(Q_final) + np.std(Q_final), color='orange', ls=':', lw=1.2)
    ax.axvline(np.mean(Q_final) - np.std(Q_final), color='orange', ls=':', lw=1.2,
               label=f'std = {np.std(Q_final):.2f}')
    ax.set_xlabel('Q')
    ax.set_ylabel('Count')
    ax.set_title('Q Distribution')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Topological Charge [{label}]', fontsize=13)
    plt.tight_layout()
    out = os.path.join(plot_dir, 'topo_charge.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"  Saved {out}")

    # ---- Plot 3: Flow evolution for selected trajectories ----
    sample_trajs = np.linspace(0, n_traj - 1, min(5, n_traj), dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for idx in sample_trajs:
        ax.plot(flow_times, E_flow[idx], lw=0.8, label=f'traj {idx}')
    ax.set_xlabel('Flow time t')
    ax.set_ylabel('E(t)')
    ax.set_title('Energy density under Wilson flow')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for idx in sample_trajs:
        ax.plot(flow_times, Q_flow[idx], lw=0.8, label=f'traj {idx}')
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Flow time t')
    ax.set_ylabel('Q')
    ax.set_title('Topological charge under Wilson flow')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Gradient Flow Evolution [{label}]', fontsize=13)
    plt.tight_layout()
    out = os.path.join(plot_dir, 'flow_evolution.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"  Saved {out}")

    # ---- Plot 4: E(t) and t^2 E(t) — scale setting ----
    E_jk = jackknife_samples_2d(E_flow)
    t2E = flow_times[np.newaxis, :]**2 * E_flow
    t2E_jk = jackknife_samples_2d(t2E)

    E_mean, E_err = jackknife_stats_2d(E_jk)
    t2E_mean, t2E_err = jackknife_stats_2d(t2E_jk)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.errorbar(flow_times, E_mean, E_err, fmt='-', lw=0.8, capsize=2,
                color='steelblue', ecolor='steelblue', alpha=0.8)
    ax.set_xlabel('Flow time t / a$^2$')
    ax.set_ylabel('E(t)')
    ax.set_title('Clover energy density E(t)')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(flow_times, t2E_mean, t2E_err, fmt='o-', markersize=2, lw=0.8,
                capsize=2, color='darkorange')
    ax.axhline(0.3, color='gray', ls='--', alpha=0.7, label='$t_0$ target = 0.3')

    t0_val = find_t0_interp(flow_times, t2E_mean)
    if not np.isnan(t0_val):
        t0_jk_vals = np.array([find_t0_interp(flow_times, t2E_jk[i])
                               for i in range(n_traj)])
        valid = ~np.isnan(t0_jk_vals)
        if valid.sum() > 0:
            t0_mean_jk, t0_err_jk = jackknife_stats(t0_jk_vals[valid])
        else:
            t0_mean_jk, t0_err_jk = t0_val, np.nan

        ax.axvline(t0_val, color='purple', ls=':', alpha=0.8,
                   label=f'$t_0/a^2$ = {t0_mean_jk:.4f} $\\pm$ {t0_err_jk:.4f}')
        sqrt_t0 = np.sqrt(t0_mean_jk)
        a_flow = SQRT8T0_PHYS / (np.sqrt(8) * sqrt_t0)
        print(f"  t0/a^2 = {t0_mean_jk:.4f} +/- {t0_err_jk:.4f}, a = {a_flow:.4f} fm")
    else:
        ax.text(0.5, 0.7, '$t_0$ not reached\nin data range',
                transform=ax.transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        print("  WARNING: t0 not found in flow time range")

    ax.set_xlabel('Flow time t / a$^2$')
    ax.set_ylabel('$t^2$ $\\langle E(t) \\rangle$')
    ax.set_title('$t^2 E(t)$ -- Scale setting')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Energy Density & $t_0$ [{label}]', fontsize=13)
    plt.tight_layout()
    out = os.path.join(plot_dir, 'energy_t2E.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"  Saved {out}")

    # ---- Plot 5: Autocorrelation of Q ----
    max_lag = min(n_traj // 2, 200)
    if max_lag < 2:
        print(f"  Skipping autocorrelation (too few trajectories)")
        return
    rho_Q = autocorrelation(Q_final, max_lag)

    tau_int = 0.5 + np.cumsum(rho_Q[1:])
    tau_est = tau_int[-1]
    for i in range(1, len(rho_Q)):
        if rho_Q[i] < 0.05:
            tau_est = tau_int[i - 1]
            break

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(np.arange(max_lag), rho_Q, '-', lw=0.8, color='teal')
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel(r'$\rho$(lag)')
    ax.set_title('Autocorrelation of Q')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(np.arange(1, max_lag), tau_int, '-', lw=0.8, color='crimson')
    ax.axhline(tau_est, color='gray', ls=':', alpha=0.5,
               label=f'$\\tau_{{int}}$ $\\approx$ {tau_est:.1f}')
    ax.set_xlabel('Window')
    ax.set_ylabel(r'$\tau_\mathrm{int}$')
    ax.set_title('Integrated Autocorrelation Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Q Autocorrelation [{label}]', fontsize=13)
    plt.tight_layout()
    out = os.path.join(plot_dir, 'autocorr_Q.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"  Saved {out}")

    # ---- Plot 6: Q at different flow times ----
    flow_check_times = [0.1, 0.3, 0.5, 1.0, 2.0]
    flow_check_times = [t for t in flow_check_times if t <= flow_times[-1]]
    flow_check_idx = [np.argmin(np.abs(flow_times - t)) for t in flow_check_times]

    fig, axes = plt.subplots(1, len(flow_check_idx),
                             figsize=(3.5 * len(flow_check_idx), 4))
    if len(flow_check_idx) == 1:
        axes = [axes]
    for i, fi in enumerate(flow_check_idx):
        ax = axes[i]
        Q_at_t = Q_flow[:, fi]
        ax.hist(Q_at_t, bins=30, color='teal', alpha=0.7, edgecolor='k', lw=0.5)
        ax.axvline(np.mean(Q_at_t), color='red', ls='--', lw=1.2)
        ax.set_xlabel('Q')
        ax.set_ylabel('Count')
        t_actual = flow_times[fi]
        ax.set_title(f't = {t_actual:.2f}\n'
                     f'<Q>={np.mean(Q_at_t):.1f}, $\\sigma$={np.std(Q_at_t):.1f}',
                     fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Q Distribution at Various Flow Times [{label}]', fontsize=12)
    plt.tight_layout()
    out = os.path.join(plot_dir, 'topo_vs_flowtime.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"  Saved {out}")

    # ---- Summary ----
    print(f"\n  --- Summary [{label}] ---")
    print(f"  <Plaq> = {plaq_vals.mean():.6f} +/- {plaq_vals.std():.6f}")
    print(f"  <Q> (t={flow_times[-1]:.1f}) = {np.mean(Q_final):.2f} +/- {np.std(Q_final):.2f}")
    print(f"  tau_int(Q) ~ {tau_est:.1f}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
datasets = find_datasets(RESULTS_ROOT)
if not datasets:
    sys.exit(f"ERROR: No datasets found under {RESULTS_ROOT}")

print(f"Found {len(datasets)} dataset(s): {[d[0] for d in datasets]}")
for label, data_dir in datasets:
    plot_dir = os.path.join(PLOT_ROOT, label)
    analyze_one(label, data_dir, plot_dir)

print("\nAll done.")
