"""
analyze_wilson_loop.py
----------------------
Diagnostic plots for Wilson loop W(r,t) data.

1. W(r,t) vs t for each r (with jackknife error bars)
2. V_eff(r,t) = -ln(W(r,t)/W(r,t+1)) vs t for each r
3. W(r,t) heatmap

Auto-discovers all lattice datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import glob
import warnings

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "../../hmc/results")
PLOT_ROOT    = os.path.join(os.path.dirname(__file__), "plots")

def find_datasets(root):
    datasets = []
    for wl in sorted(glob.glob(os.path.join(root, "*/*/flow/WS_loop.csv"))):
        data_dir = os.path.dirname(wl)
        parts = data_dir.replace(root, "").strip("/").split("/")
        label = f"{parts[0]}/{parts[1]}"
        datasets.append((label, data_dir))
    return datasets

def jackknife_mean_err(data):
    """Jackknife mean and error for 1D array."""
    n = len(data)
    if n < 2:
        return data.mean(), 0.0
    jk = (data.sum() - data) / (n - 1)
    mean = jk.mean()
    err = np.sqrt((n - 1) * np.var(jk, ddof=0))
    return mean, err

def analyze_one(label, data_dir, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    wl_file = os.path.join(data_dir, "WS_loop.csv")
    df = pd.read_csv(wl_file)
    print(f"\n[{label}] Loaded {wl_file}: {len(df)} rows")

    r_vals = np.sort(df["r"].unique()).astype(int)
    t_vals = np.sort(df["t"].unique()).astype(int)
    n_traj = df["traj"].nunique()
    print(f"  Trajectories: {n_traj}, r: 1..{r_vals[-1]}, t: 1..{t_vals[-1]}")

    # Build W_data[traj, r_idx, t_idx]
    W_data = np.full((n_traj, len(r_vals), len(t_vals)), np.nan)
    r_map = {r: i for i, r in enumerate(r_vals)}
    t_map = {t: i for i, t in enumerate(t_vals)}
    for _, row in df.iterrows():
        W_data[int(row["traj"]), r_map[int(row["r"])], t_map[int(row["t"])]] = row["W"]

    # Compute jackknife mean and error
    W_mean = np.zeros((len(r_vals), len(t_vals)))
    W_err  = np.zeros((len(r_vals), len(t_vals)))
    for ri in range(len(r_vals)):
        for ti in range(len(t_vals)):
            vals = W_data[:, ri, ti]
            good = ~np.isnan(vals)
            if good.sum() >= 2:
                W_mean[ri, ti], W_err[ri, ti] = jackknife_mean_err(vals[good])
            elif good.sum() == 1:
                W_mean[ri, ti] = vals[good].mean()

    # ---- Plot 1: W(r,t) vs t for selected r values ----
    r_show = r_vals[:min(8, len(r_vals))]
    ncol = min(4, len(r_show))
    nrow = (len(r_show) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 3.5*nrow))
    if nrow * ncol == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, r in zip(axes, r_show):
        ri = r_map[r]
        ax.errorbar(t_vals, W_mean[ri], yerr=W_err[ri],
                    fmt="o-", ms=3, capsize=2, lw=1, color="steelblue")
        ax.set_title(f"r = {r}")
        ax.set_xlabel("t")
        ax.set_ylabel("W(r,t)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        # Mark where W goes negative or zero
        neg = W_mean[ri] <= 0
        if neg.any():
            ax.axhline(0, color="red", ls=":", lw=0.8)

    for ax in axes[len(r_show):]:
        ax.set_visible(False)

    fig.suptitle(f"Wilson loop W(r,t) vs t [{label}]", fontsize=13)
    fig.tight_layout()
    out = os.path.join(plot_dir, "wilson_loop_vs_t.png")
    fig.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.close(fig)

    # ---- Plot 2: W(r,t) vs r for selected t values ----
    t_show_vals = [1, 2, 3, 4, 6, 8]
    t_show_vals = [t for t in t_show_vals if t in t_map]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(t_show_vals)))
    for color, t in zip(colors, t_show_vals):
        ti = t_map[t]
        ax.errorbar(r_vals, W_mean[:, ti], yerr=W_err[:, ti],
                    fmt="o-", ms=3, capsize=2, lw=1, color=color, label=f"t={t}")
    ax.set_xlabel("r")
    ax.set_ylabel("W(r,t)")
    ax.set_title(f"Wilson loop vs r [{label}]")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(plot_dir, "wilson_loop_vs_r.png")
    fig.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.close(fig)

    # ---- Plot 3: Heatmap of W(r,t) ----
    fig, ax = plt.subplots(figsize=(8, 6))
    # Use log scale for heatmap, clip negatives
    W_plot = W_mean.copy()
    W_plot[W_plot <= 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        W_log = np.log10(W_plot)
    im = ax.imshow(W_log, aspect="auto", origin="lower",
                   extent=[t_vals[0]-0.5, t_vals[-1]+0.5,
                           r_vals[0]-0.5, r_vals[-1]+0.5],
                   cmap="viridis")
    ax.set_xlabel("t")
    ax.set_ylabel("r")
    ax.set_title(f"log10 W(r,t) [{label}]")
    plt.colorbar(im, ax=ax, label="log10 W")
    fig.tight_layout()
    out = os.path.join(plot_dir, "wilson_loop_heatmap.png")
    fig.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.close(fig)

    # ---- Plot 4: V_eff(r,t) = -ln(W(r,t)/W(r,t+1)) ----
    r_show2 = r_vals[:min(8, len(r_vals))]
    fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 3.5*nrow))
    if nrow * ncol == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, r in zip(axes, r_show2):
        ri = r_map[r]
        t_mid = t_vals[:-1]
        Veff = np.full(len(t_mid), np.nan)
        Veff_err = np.full(len(t_mid), np.nan)
        for k in range(len(t_mid)):
            num = W_data[:, ri, k]
            den = W_data[:, ri, k+1]
            mask = (num > 0) & (den > 0) & ~np.isnan(num) & ~np.isnan(den)
            if mask.sum() >= 2:
                n = mask.sum()
                jk_num = (num[mask].sum() - num[mask]) / (n - 1)
                jk_den = (den[mask].sum() - den[mask]) / (n - 1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    jk_v = -np.log(jk_num / jk_den)
                Veff[k] = jk_v.mean()
                Veff_err[k] = np.sqrt((n - 1) * np.var(jk_v, ddof=0))

        good = ~np.isnan(Veff)
        if good.any():
            ax.errorbar(t_mid[good], Veff[good], yerr=Veff_err[good],
                        fmt="o-", ms=3, capsize=2, color="steelblue")
        ax.set_title(f"r = {r}")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$V_\mathrm{eff}(r,t)$")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(r_show2):]:
        ax.set_visible(False)

    fig.suptitle(f"Effective potential V_eff(r,t) [{label}]", fontsize=13)
    fig.tight_layout()
    out = os.path.join(plot_dir, "wilson_loop_Veff.png")
    fig.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.close(fig)

    # ---- Print raw W values for small r,t as sanity check ----
    print(f"\n  W(r,t) sanity check (first few r,t):")
    print(f"  {'r':>3} {'t':>3} {'W_mean':>12} {'W_err':>12}")
    for r in r_vals[:4]:
        for t in t_vals[:6]:
            ri, ti = r_map[r], t_map[t]
            print(f"  {r:3d} {t:3d} {W_mean[ri,ti]:12.6f} {W_err[ri,ti]:12.6f}")

# ---------------------------------------------------------------------------
datasets = find_datasets(RESULTS_ROOT)
if not datasets:
    sys.exit(f"ERROR: No datasets found under {RESULTS_ROOT}")

print(f"Found {len(datasets)} dataset(s): {[d[0] for d in datasets]}")
for label, data_dir in datasets:
    plot_dir = os.path.join(PLOT_ROOT, label)
    analyze_one(label, data_dir, plot_dir)

print("\nDone.")
