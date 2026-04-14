"""
analyze_sommer.py
-----------------
Sommer parameter r0 from the static quark potential V(r).

Method:
  1. Load W(r,t) from WS_loop.csv
  2. For each r: fit W(r,t) = A(r) * exp(-V(r) * t) with auto t-range selection
  3. Jackknife the full chain: W → V(r) → Cornell fit → r₀
  4. Sommer condition: r0^2 * dV/dr|_{r0} = 1.65
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import os
import sys
import glob
import warnings

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "../../hmc/results")
PLOT_ROOT    = os.path.join(os.path.dirname(__file__), "plots")

SOMMER_C   = 1.65
R0_PHYS_FM = 0.5

SNR_CUT    = 3.0
T_MIN      = 1
T_FIT_MIN_PTS = 3
R_MAX_FRAC = 0.5     # only use r <= Ls * R_MAX_FRAC to avoid wraparound

def find_datasets(root):
    datasets = []
    for wl in sorted(glob.glob(os.path.join(root, "*/*/flow/WS_loop.csv"))):
        data_dir = os.path.dirname(wl)
        parts = data_dir.replace(root, "").strip("/").split("/")
        label = f"{parts[0]}/{parts[1]}"
        datasets.append((label, data_dir))
    return datasets

def cornell(r, V0, sigma, e):
    return V0 + sigma * r - e / r

def fit_V_single_r(W_at_r_t, t_vals, W_err_t=None):
    """
    Fit W(t) = A * exp(-V*t) for a single r value.
    W_at_r_t: 1D array of W values at different t
    W_err_t: optional 1D array of errors (for SNR cut on central fit)
    Returns (V, A, t_fit) or (nan, nan, []) if fit fails
    """
    W_mean = W_at_r_t
    usable = np.zeros(len(t_vals), dtype=bool)
    for ti in range(len(t_vals)):
        if t_vals[ti] < T_MIN or W_mean[ti] <= 0:
            continue
        if W_err_t is not None and W_err_t[ti] > 0:
            if W_mean[ti] / W_err_t[ti] < SNR_CUT:
                continue
        usable[ti] = True

    t_indices = []
    started = False
    for ti in range(len(t_vals)):
        if t_vals[ti] < T_MIN:
            continue
        if usable[ti]:
            t_indices.append(ti)
            started = True
        elif started:
            break

    if len(t_indices) < T_FIT_MIN_PTS:
        return np.nan, np.nan, []

    t_fit = t_vals[t_indices].astype(float)
    W_fit = W_mean[t_indices]

    if np.any(W_fit <= 0):
        return np.nan, np.nan, []

    # Initial guess from log-linear
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_W = np.log(np.maximum(W_fit, 1e-30))
    try:
        p = np.polyfit(t_fit, log_W, 1)
        V_guess = -p[0]
        A_guess = np.exp(p[1])
    except:
        return np.nan, np.nan, []

    # Correlated fit using errors as weights if available
    if W_err_t is not None:
        err_fit = W_err_t[t_indices]
        if np.all(err_fit > 0):
            try:
                def model(t, A, V):
                    return A * np.exp(-V * t)
                popt, _ = curve_fit(model, t_fit, W_fit, sigma=err_fit,
                                    p0=[A_guess, V_guess], maxfev=5000)
                return popt[1], popt[0], t_fit
            except:
                pass

    # Fallback: log-linear
    return V_guess, A_guess, t_fit

def analyze_one(label, data_dir, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    wl_file = os.path.join(data_dir, "WS_loop.csv")
    df = pd.read_csv(wl_file)
    print(f"\n[{label}] Loaded {wl_file}: {len(df)} rows")

    r_vals = np.sort(df["r"].unique()).astype(int)
    t_vals = np.sort(df["t"].unique()).astype(int)
    n_traj = df["traj"].nunique()

    # Cut r to avoid wraparound: use r <= Ls/2
    Ls = r_vals[-1]  # Ls = max r = Lx
    r_max = int(Ls * R_MAX_FRAC)
    r_vals = r_vals[r_vals <= r_max]
    print(f"  Trajectories: {n_traj}, r: 1..{r_vals[-1]} (max {Ls}, cut at {r_max}), t: 1..{t_vals[-1]}")

    # Filter DataFrame to only keep r <= r_max
    df = df[df["r"].isin(r_vals)]

    # Build W_data[traj, r_idx, t_idx]
    W_data = np.full((n_traj, len(r_vals), len(t_vals)), np.nan)
    r_map = {r: i for i, r in enumerate(r_vals)}
    t_map = {t: i for i, t in enumerate(t_vals)}
    for _, row in df.iterrows():
        W_data[int(row["traj"]), r_map[int(row["r"])], t_map[int(row["t"])]] = row["W"]

    # ---------------------------------------------------------------
    # Jackknife the full chain: W → V(r) → Cornell → r₀
    # ---------------------------------------------------------------
    n = n_traj

    def extract_V_array(W_sample, W_err=None):
        """
        Given W_sample[r_idx, t_idx], extract V(r) for all r.
        W_err: optional error array for SNR cut (central fit only).
        Returns V_arr, A_arr, t_ranges_dict
        """
        V_arr = np.full(len(r_vals), np.nan)
        A_arr = np.full(len(r_vals), np.nan)
        t_ranges = {}
        for ri in range(len(r_vals)):
            err_r = W_err[ri, :] if W_err is not None else None
            V, A, t_fit = fit_V_single_r(W_sample[ri, :], t_vals, err_r)
            V_arr[ri] = V
            A_arr[ri] = A
            t_ranges[r_vals[ri]] = t_fit
        return V_arr, A_arr, t_ranges

    def fit_cornell_and_r0(V_arr):
        """
        Fit Cornell to V(r), compute r₀.
        Returns (V0, sigma, e, r0) or (nan, nan, nan, nan) if fit fails
        """
        good = ~np.isnan(V_arr)
        if good.sum() < 3:
            return np.nan, np.nan, np.nan, np.nan

        r_fit = r_vals[good].astype(float)
        V_fit = V_arr[good]

        try:
            popt, _ = curve_fit(cornell, r_fit, V_fit, p0=[0.0, 0.2, 0.3], maxfev=10000)
            V0, sigma, e = popt
            # Sommer: r0^2 (sigma + e/r0^2) = 1.65
            arg = (SOMMER_C - e) / sigma
            if sigma > 0 and arg > 0:
                r0 = np.sqrt(arg)
                return V0, sigma, e, r0
        except:
            pass
        return np.nan, np.nan, np.nan, np.nan

    # Central values with SNR cut
    W_mean = np.nanmean(W_data, axis=0)  # (n_r, n_t)

    # Compute errors for SNR cut
    W_err_central = np.zeros((len(r_vals), len(t_vals)))
    for ri in range(len(r_vals)):
        for ti in range(len(t_vals)):
            col = W_data[:, ri, ti]
            good = ~np.isnan(col)
            if good.sum() >= 2:
                jk = (col[good].sum() - col[good]) / (good.sum() - 1)
                W_err_central[ri, ti] = np.sqrt((good.sum() - 1) * np.var(jk, ddof=0))

    V_central, A_central, t_ranges_central = extract_V_array(W_mean, W_err_central)
    V0_c, sigma_c, e_c, r0_c = fit_cornell_and_r0(V_central)

    print(f"\n  Central fit V(r) extraction:")
    print(f"  {'r':>3} {'V(r)':>10} {'t_range':>15} {'status':>8}")
    for ri, r in enumerate(r_vals):
        t_fit = t_ranges_central.get(r, [])
        t_str = f"[{int(t_fit[0])},{int(t_fit[-1])}]({len(t_fit)})" if len(t_fit) > 0 else "N/A"
        status = "OK" if not np.isnan(V_central[ri]) else "FAIL"
        if status == "OK":
            print(f"  {r:3d} {V_central[ri]:10.4f} {t_str:>15} {status:>8}")
        else:
            print(f"  {r:3d} {'---':>10} {t_str:>15} {status:>8}")

    # Jackknife samples: use FIXED t-ranges from central fit
    V_jk_arr = np.zeros((n, len(r_vals)))   # V(r) per jackknife sample
    V0_jk = np.zeros(n)
    sigma_jk = np.zeros(n)
    e_jk = np.zeros(n)
    r0_jk = np.zeros(n)

    for j in range(n):
        # Jackknife sample: average over all except j
        W_jk = (W_data.sum(axis=0) - W_data[j]) / (n - 1)

        # Refit V(r) using the same t-ranges as central fit
        V_jk_r = np.full(len(r_vals), np.nan)
        for ri in range(len(r_vals)):
            t_fit = t_ranges_central.get(r_vals[ri], [])
            if len(t_fit) < T_FIT_MIN_PTS:
                continue
            t_idx = np.array([t_map[int(t)] for t in t_fit])
            W_fit = W_jk[ri, t_idx]
            t_f = t_fit.astype(float)
            if np.any(W_fit <= 0):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                log_W = np.log(np.maximum(W_fit, 1e-30))
            try:
                p = np.polyfit(t_f, log_W, 1)
                V_jk_r[ri] = -p[0]
            except:
                pass

        V_jk_arr[j] = V_jk_r
        V0_jk[j], sigma_jk[j], e_jk[j], r0_jk[j] = fit_cornell_and_r0(V_jk_r)

    # Jackknife errors
    valid = ~np.isnan(r0_jk)
    if valid.sum() >= 2:
        V0_err = np.sqrt((n - 1) * np.var(V0_jk[valid], ddof=0))
        sigma_err = np.sqrt((n - 1) * np.var(sigma_jk[valid], ddof=0))
        e_err = np.sqrt((n - 1) * np.var(e_jk[valid], ddof=0))
        r0_err = np.sqrt((n - 1) * np.var(r0_jk[valid], ddof=0))
    else:
        V0_err = sigma_err = e_err = r0_err = np.nan

    if not np.isnan(r0_c):
        a_fm = R0_PHYS_FM / r0_c
        a_fm_err = a_fm * r0_err / r0_c if not np.isnan(r0_err) else np.nan

        print(f"\n  --- Cornell fit: V(r) = V0 + sigma*r - e/r ---")
        print(f"  V0    = {V0_c:.4f} +/- {V0_err:.4f}")
        print(f"  sigma = {sigma_c:.6f} +/- {sigma_err:.6f}")
        print(f"  e     = {e_c:.4f} +/- {e_err:.4f}")
        print(f"\n  --- Sommer parameter ---")
        print(f"  r0/a = {r0_c:.4f} +/- {r0_err:.4f}")
        print(f"  a    = {a_fm:.4f} +/- {a_fm_err:.4f} fm")
    else:
        print(f"\n  WARNING: Cornell fit or r0 extraction failed")

    # ---------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------

    # Plot 1: W(r,t) with fit overlay
    r_show = r_vals[:min(8, len(r_vals))]
    ncol = min(4, len(r_show))
    nrow = (len(r_show) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 3.5*nrow))
    if nrow * ncol == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, r in zip(axes, r_show):
        ri = r_map[r]
        W_mean_all = np.nanmean(W_data[:, ri, :], axis=0)
        W_err_all = np.zeros(len(t_vals))
        for ti in range(len(t_vals)):
            col = W_data[:, ri, ti]
            gd = ~np.isnan(col)
            if gd.sum() >= 2:
                jk = (col[gd].sum() - col[gd]) / (gd.sum() - 1)
                W_err_all[ti] = np.sqrt((gd.sum() - 1) * np.var(jk, ddof=0))

        t_in_fit = set(t_ranges_central.get(r, []).astype(int)) if len(t_ranges_central.get(r, [])) > 0 else set()
        in_fit = np.array([t in t_in_fit for t in t_vals])
        out_fit = ~in_fit & (W_mean_all > 0)

        if out_fit.any():
            ax.errorbar(t_vals[out_fit], W_mean_all[out_fit], yerr=W_err_all[out_fit],
                        fmt="o", ms=3, capsize=2, color="gray", alpha=0.4, label="excluded")
        if in_fit.any():
            ax.errorbar(t_vals[in_fit], W_mean_all[in_fit], yerr=W_err_all[in_fit],
                        fmt="o", ms=4, capsize=2, color="steelblue", label="used in fit")

        if not np.isnan(V_central[ri]) and not np.isnan(A_central[ri]):
            t_dense = np.linspace(t_vals[in_fit][0], t_vals[in_fit][-1], 100) if in_fit.any() else []
            if len(t_dense) > 0:
                ax.plot(t_dense, A_central[ri] * np.exp(-V_central[ri] * t_dense),
                        "r-", lw=1.5, label=f"V={V_central[ri]:.3f}")

        ax.set_title(f"r = {r}")
        ax.set_xlabel("t")
        ax.set_ylabel("W(r,t)")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(r_show):]:
        ax.set_visible(False)

    fig.suptitle(f"Wilson loop fit: W(r,t) = A exp(-Vt) [{label}]", fontsize=13)
    fig.tight_layout()
    out = os.path.join(plot_dir, "sommer_wloop_fit.png")
    fig.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.close(fig)

    # Plot 2: V(r) with Cornell fit and jackknife error band
    fig, ax = plt.subplots(figsize=(8, 5))

    # Jackknife V(r) error bars (already computed above)
    V_err_arr = np.zeros(len(r_vals))
    for ri in range(len(r_vals)):
        valid_ri = ~np.isnan(V_jk_arr[:, ri])
        if valid_ri.sum() >= 2:
            V_err_arr[ri] = np.sqrt((n - 1) * np.var(V_jk_arr[valid_ri, ri], ddof=0))

    good = ~np.isnan(V_central) & ~np.isnan(V_err_arr) & (V_err_arr > 0)
    ax.errorbar(r_vals[good], V_central[good], yerr=V_err_arr[good],
                fmt="o", ms=6, capsize=4, color="steelblue", label="V(r) from fit")

    if not np.isnan(r0_c):
        r_dense = np.linspace(max(0.5, r_vals[good][0] * 0.8), r_vals[good][-1] * 1.1, 200)
        ax.plot(r_dense, cornell(r_dense, V0_c, sigma_c, e_c), "r-", lw=1.8,
                label=rf"Cornell: $\sigma$={sigma_c:.4f}, $e$={e_c:.3f}")

        V_r0 = cornell(r0_c, V0_c, sigma_c, e_c)
        ax.axvline(r0_c, color="green", lw=1.2, ls="--",
                   label=f"$r_0/a$={r0_c:.2f}")
        ax.plot(r0_c, V_r0, "g^", ms=10)

    ax.set_xlabel("r / a")
    ax.set_ylabel("V(r)")
    ax.set_title(f"Static quark potential [{label}]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(plot_dir, "sommer_potential.png")
    fig.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.close(fig)

    # Plot 3: V_eff plateau check
    fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 3.5*nrow))
    if nrow * ncol == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, r in zip(axes, r_show):
        ri = r_map[r]
        t_mid = t_vals[:-1]
        Veff = np.full(len(t_mid), np.nan)
        Veff_err = np.full(len(t_mid), np.nan)
        for k in range(len(t_mid)):
            num = W_data[:, ri, k]
            den = W_data[:, ri, k+1]
            mask = ~np.isnan(num) & ~np.isnan(den) & (num > 0) & (den > 0)
            if mask.sum() >= 2:
                jk_num = (num[mask].sum() - num[mask]) / (mask.sum() - 1)
                jk_den = (den[mask].sum() - den[mask]) / (mask.sum() - 1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    jk_v = -np.log(jk_num / jk_den)
                good_jk = np.isfinite(jk_v)
                if good_jk.sum() >= 2:
                    Veff[k] = jk_v[good_jk].mean()
                    Veff_err[k] = np.sqrt((mask.sum() - 1) * np.var(jk_v[good_jk], ddof=0))

        ok = np.isfinite(Veff) & np.isfinite(Veff_err)
        if ok.any():
            ax.errorbar(t_mid[ok], Veff[ok], yerr=Veff_err[ok],
                        fmt="o-", ms=3, capsize=2, color="steelblue")

        if not np.isnan(V_central[ri]):
            ax.axhline(V_central[ri], color="red", lw=1.2, ls="--",
                       label=f"V={V_central[ri]:.3f}")

        t_fit_r = t_ranges_central.get(r, np.array([]))
        if len(t_fit_r) > 0:
            ax.axvspan(t_fit_r[0], t_fit_r[-1], alpha=0.1, color="green")

        ax.set_title(f"r = {r}")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$V_\mathrm{eff}(r,t)$")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(r_show):]:
        ax.set_visible(False)

    fig.suptitle(f"Effective mass plateau check [{label}]", fontsize=13)
    fig.tight_layout()
    out = os.path.join(plot_dir, "sommer_plateau.png")
    fig.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.close(fig)

# ---------------------------------------------------------------------------
datasets = find_datasets(RESULTS_ROOT)
if not datasets:
    sys.exit(f"ERROR: No datasets found under {RESULTS_ROOT}")

print(f"Found {len(datasets)} dataset(s): {[d[0] for d in datasets]}")
for label, data_dir in datasets:
    plot_dir = os.path.join(PLOT_ROOT, label)
    analyze_one(label, data_dir, plot_dir)

print("\nDone.")
