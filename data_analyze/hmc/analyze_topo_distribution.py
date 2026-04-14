"""
analyze_topo_distribution.py
----------------------------
Topological charge Q distribution as a function of gradient flow time.

Auto-discovers all lattice datasets under results/.
Outputs plots to plots/{dataset}/ subdirectories.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import glob

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "../../hmc/results")
PLOT_ROOT    = os.path.join(os.path.dirname(__file__), "plots")

def find_datasets(root):
    """Find all directories containing smear.csv under results root."""
    datasets = []
    for smear in sorted(glob.glob(os.path.join(root, "*/*/flow/smear.csv"))):
        data_dir = os.path.dirname(smear)
        parts = data_dir.replace(root, "").strip("/").split("/")
        label = f"{parts[0]}/{parts[1]}"  # e.g. T32_S16/beta6.0
        datasets.append((label, data_dir))
    return datasets

def analyze_one(label, data_dir, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    smear_file = os.path.join(data_dir, "smear.csv")
    df = pd.read_csv(smear_file)
    print(f"\n[{label}] Loaded {smear_file}: {len(df)} rows")

    t_vals  = np.sort(df["t_flow"].unique())
    n_traj  = df["traj"].nunique()
    n_steps = len(t_vals)
    print(f"  Trajectories: {n_traj},  flow steps: {n_steps}")
    print(f"  t_flow range: [{t_vals[0]:.4f}, {t_vals[-1]:.4f}]")

    Q_pivot = df.pivot(index="traj", columns="t_flow", values="Q").values

    # ---- Plot 1: Q histograms at selected flow times ----
    n_show   = min(6, n_steps)
    idx_show = np.round(np.linspace(0, n_steps - 1, n_show)).astype(int)
    t_show   = t_vals[idx_show]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for ax, ti in zip(axes, t_show):
        col_idx = np.where(t_vals == ti)[0][0]
        Q_at_t  = Q_pivot[:, col_idx]
        Q_mean  = Q_at_t.mean()
        Q_std   = Q_at_t.std()

        bins = np.arange(np.floor(Q_at_t.min()) - 0.5,
                         np.ceil(Q_at_t.max())  + 1.5, 0.1)
        ax.hist(Q_at_t, bins=bins, color="steelblue", edgecolor="none", alpha=0.85)
        ax.axvline(Q_mean, color="red",    lw=1.5, ls="--", label=f"mean={Q_mean:.2f}")
        ax.axvline(0,      color="black",  lw=0.8, ls=":")
        ax.set_title(f"t_flow = {ti:.3f}")
        ax.set_xlabel("Q")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.text(0.97, 0.95, f"$\\sigma$={Q_std:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8)

    for ax in axes[len(t_show):]:
        ax.set_visible(False)

    fig.suptitle(f"Topological charge distribution [{label}]", fontsize=13)
    fig.tight_layout()
    out1 = os.path.join(plot_dir, "topo_distribution.png")
    fig.savefig(out1, dpi=150)
    print(f"  Saved {out1}")
    plt.close(fig)

    # ---- Plot 2: Mean and std of Q vs flow time ----
    Q_mean_vs_t = Q_pivot.mean(axis=0)
    Q_std_vs_t  = Q_pivot.std(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(t_vals, Q_mean_vs_t, "o-", ms=3, lw=1.2, color="steelblue")
    ax1.axhline(0, color="black", lw=0.8, ls="--")
    ax1.set_xlabel("flow time t")
    ax1.set_ylabel(r"$\langle Q \rangle$")
    ax1.set_title(f"Mean Q vs flow time [{label}]")

    ax2.plot(t_vals, Q_std_vs_t, "s-", ms=3, lw=1.2, color="darkorange")
    ax2.set_xlabel("flow time t")
    ax2.set_ylabel(r"$\sigma_Q$")
    ax2.set_title(f"Std of Q vs flow time [{label}]")

    fig.tight_layout()
    out2 = os.path.join(plot_dir, "topo_std_vs_flow.png")
    fig.savefig(out2, dpi=150)
    print(f"  Saved {out2}")
    plt.close(fig)

    # ---- Summary ----
    print(f"  Summary at t={t_vals[-1]:.3f}: <Q>={Q_mean_vs_t[-1]:.4f}, "
          f"sigma(Q)={Q_std_vs_t[-1]:.4f}")

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

print("\nDone.")
