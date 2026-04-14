"""
analyze_scale_t0.py
-------------------
Extract the gradient flow scale t0 from t^2 <E(t)> = 0.3.

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

T0_TARGET    = 0.3          # Luscher t0 condition: t^2 E(t0) = 0.3
SQRT8T0_PHYS = 0.415        # fm;  physical reference sqrt(8 t0)|_phys ~ 0.415 fm

# ---------------------------------------------------------------------------
# Jackknife utilities
# ---------------------------------------------------------------------------
def jackknife_array(arr2d):
    n      = arr2d.shape[0]
    jk     = (arr2d.sum(axis=0) - arr2d) / (n - 1)
    mean   = jk.mean(axis=0)
    err    = np.sqrt((n - 1) * jk.var(axis=0, ddof=0))
    return mean, err

def find_datasets(root):
    datasets = []
    for smear in sorted(glob.glob(os.path.join(root, "*/*/flow/smear.csv"))):
        data_dir = os.path.dirname(smear)
        parts = data_dir.replace(root, "").strip("/").split("/")
        label = f"{parts[0]}/{parts[1]}"
        datasets.append((label, data_dir))
    return datasets

# ---------------------------------------------------------------------------
# Analyze one dataset
# ---------------------------------------------------------------------------
def analyze_one(label, data_dir, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    smear_file = os.path.join(data_dir, "smear.csv")
    df = pd.read_csv(smear_file)
    print(f"\n[{label}] Loaded {smear_file}: {len(df)} rows")

    t_vals = np.sort(df["t_flow"].unique())
    n_traj = df["traj"].nunique()
    print(f"  Trajectories: {n_traj},  flow steps: {len(t_vals)}")

    E_pivot = df.pivot(index="traj", columns="t_flow", values="E").values
    t2E_pivot = (t_vals**2)[np.newaxis, :] * E_pivot

    t2E_mean, t2E_err = jackknife_array(t2E_pivot)
    E_mean,   E_err   = jackknife_array(E_pivot)

    # Find t0
    if t2E_mean.max() < T0_TARGET:
        print(f"  WARNING: t^2<E> never reaches {T0_TARGET}. "
              f"Max = {t2E_mean.max():.4f}. Increase flow_t_max.")
        t0_over_a2 = np.nan
        a_fm = np.nan
        t0_err = np.nan
    else:
        f_interp = interp1d(t2E_mean, t_vals, kind="linear")
        t0_over_a2 = float(f_interp(T0_TARGET))

        n = n_traj
        jk_t2E = (t2E_pivot.sum(axis=0) - t2E_pivot) / (n - 1)
        t0_jk  = np.zeros(n)
        for i in range(n):
            row = jk_t2E[i]
            if row.max() >= T0_TARGET:
                t0_jk[i] = float(interp1d(row, t_vals, kind="linear")(T0_TARGET))
            else:
                t0_jk[i] = np.nan
        valid = ~np.isnan(t0_jk)
        t0_err = np.sqrt((n - 1) * np.var(t0_jk[valid], ddof=0))

        a_fm = SQRT8T0_PHYS / np.sqrt(8.0 * t0_over_a2)
        a_fm_err = 0.5 * a_fm * t0_err / t0_over_a2

        print(f"  --- Scale setting ---")
        print(f"    t0/a^2  = {t0_over_a2:.4f} +/- {t0_err:.4f}")
        print(f"    a       = {a_fm:.4f} +/- {a_fm_err:.4f} fm")

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.fill_between(t_vals, E_mean - E_err, E_mean + E_err,
                     alpha=0.3, color="steelblue", label="jackknife error")
    ax1.plot(t_vals, E_mean, "o-", ms=2.5, lw=1.5, color="steelblue",
             label=r"$\langle E(t) \rangle$")
    n_show = min(5, n_traj)
    for i in range(n_show):
        ax1.plot(t_vals, E_pivot[i], lw=0.5, alpha=0.4, color="gray")
    ax1.set_xlabel("flow time t")
    ax1.set_ylabel("E(t)")
    ax1.set_title(f"Clover energy density [{label}]")
    ax1.legend(fontsize=9)

    ax2.fill_between(t_vals, t2E_mean - t2E_err, t2E_mean + t2E_err,
                     alpha=0.3, color="darkorange")
    ax2.plot(t_vals, t2E_mean, "s-", ms=2.5, lw=1.5, color="darkorange",
             label=r"$t^2 \langle E(t) \rangle$")
    ax2.axhline(T0_TARGET, color="red", lw=1.2, ls="--", label=f"target = {T0_TARGET}")

    if not np.isnan(t0_over_a2):
        ax2.axvline(t0_over_a2, color="green", lw=1.2, ls="--",
                    label=f"$t_0/a^2$ = {t0_over_a2:.3f}")
        ax2.annotate(
            f"$t_0/a^2={t0_over_a2:.3f}$\n$a={a_fm:.3f}$ fm",
            xy=(t0_over_a2, T0_TARGET),
            xytext=(t0_over_a2 * 1.3, T0_TARGET * 0.8),
            fontsize=9, color="green",
            arrowprops=dict(arrowstyle="->", color="green", lw=0.8)
        )

    ax2.set_xlabel("flow time t")
    ax2.set_ylabel(r"$t^2 E(t)$")
    ax2.set_title(f"Scale setting [{label}]")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(plot_dir, "scale_t0.png")
    fig.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.close(fig)

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
