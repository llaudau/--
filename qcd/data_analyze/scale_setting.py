#!/usr/bin/env python3
"""
Scale Setting Comparison: Wilson Loop vs Gradient Flow

Two independent methods to extract the lattice spacing a:

Method 1 — Static Quark Potential (Wilson loops):
  W(r,t) = A(r) * exp(-V(r)*t)
  V(r) = -A/r + sigma*r + C  (Cornell potential)
  Sommer parameter: r0 defined by r0^2 * F(r0) = 1.65, F = dV/dr
  Physical: r0 ~ 0.5 fm  =>  a = r0_lattice * (0.5 fm / r0_phys) [fm/a is not computed here,
  only the dimensionless ratio r0/a is extracted]

Method 2 — Yang-Mills Gradient Flow (t0 scale):
  E(t) = 6.0 - plaquette  (energy density from gauge_wilson_flow)
  t0 defined by:  t0^2 * <E(t0)> = 0.3
  Physical: sqrt(8*t0) ~ 0.415 fm

Consistency check:
  r0/a  and  sqrt(t0)/a  can both be converted to a (fm).
  They should agree.

Data files (from autocorrelate_hmc.cu output):
  wilson_loop_data.txt    : T*S*n_samples values, layout [t_ext, x_ext, traj]
  energy_density_data.txt : smear_steps*n_samples values, layout [smear, traj]
  flow_times.txt          : smear_steps values (flow time at each smear step)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, brentq
import os

# ─── Configuration ────────────────────────────────────────────────────────────
T = 16
S = 8
beta = 5.5
n_samples = 10000

data_dir   = f"/home/khw/Documents/Git_repository/hmc_gpu/hmc_results/t{T}_s{S}_beta{beta:.1f}"
output_dir = f"/home/khw/Documents/Git_repository/qcd/data_analyze/result4/T{T}_S{S}_beta{beta:.1f}"
os.makedirs(output_dir, exist_ok=True)

# Physical reference values
r0_phys_fm     = 0.5          # fm
sqrt8t0_phys_fm = 0.415       # fm  (sqrt(8*t0) ~ 0.415 fm)

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")

wilson_file = os.path.join(data_dir, "wilson_loop_data.txt")
energy_file = os.path.join(data_dir, "energy_density_data.txt")
ft_file     = os.path.join(data_dir, "flow_times.txt")

wilson_raw = np.loadtxt(wilson_file)           # T*S*n_samples values
energy_raw = np.loadtxt(energy_file)           # smear_steps*n_samples values
flow_times = np.loadtxt(ft_file)               # smear_steps values
smear_steps = len(flow_times)                  # auto-detect smear_steps

# Reshape Wilson loop data: [t_ext, x_ext, configs] → [configs, t_ext, x_ext]
wilson_data = wilson_raw.reshape(T, S, n_samples)
wilson_data = np.transpose(wilson_data, (2, 0, 1))   # [configs, T, S]
print(f"  Wilson loops: {wilson_data.shape}  (configs, t_ext, x_ext)")

# Reshape energy density: [smear_steps, configs]
energy_data = energy_raw.reshape(smear_steps, n_samples)
print(f"  Energy density: {energy_data.shape}  (smear_steps, configs)")
print(f"  Flow times: {flow_times}")

# ─── Jackknife helper ─────────────────────────────────────────────────────────
def jackknife_1d(data):
    """data shape: [configs]. Returns jackknife samples [configs]."""
    n = len(data)
    mean = np.mean(data)
    return (mean * n - data) / (n - 1)

def jackknife_nd(data):
    """data shape: [configs, ...]. Returns jackknife samples [configs, ...]."""
    n = data.shape[0]
    mean = np.mean(data, axis=0)
    return (mean[np.newaxis] * n - data) / (n - 1)

def jackknife_stats(jk_samples):
    """Returns (mean, error) from jackknife samples."""
    n = len(jk_samples)
    mean = np.mean(jk_samples, axis=0)
    err  = np.sqrt((n - 1) * np.mean((jk_samples - mean)**2, axis=0))
    return mean, err

# ═══════════════════════════════════════════════════════════════════════════════
# METHOD 1: Wilson loop → static potential → Sommer parameter r0
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("METHOD 1: Wilson Loop → Static Potential")
print("="*60)

def wilson_exp(t, A, V):
    return A * np.exp(-V * t)

def cornell(r, A_c, sigma, C):
    return -A_c / r + sigma * r + C

# Spatial separations and time values (1-indexed, avoid boundary)
# Use r = 1 .. S-1 (exclude r=0 trivial and r=S identical to r=0 by PBC)
r_max  = S - 1           # e.g. S=10 → r_max=9  (r=1..9)
r_vals = np.arange(1, r_max + 1, dtype=float)
t_vals = np.arange(1, T, dtype=float)     # skip t=0
print(f"  Using r = 1..{r_max} (exclude r=0 and r=S={S} boundary)")

# --- Jackknife fit of W(r,t) → V(r) ---
print("Fitting W(r,t) = A*exp(-V*t) for each r via jackknife...")

wilson_jk = jackknife_nd(wilson_data)    # [configs, T, S]

V_jk = np.zeros((n_samples, r_max))
A_jk = np.zeros((n_samples, r_max))

for jk in range(n_samples):
    if jk % 2000 == 0:
        print(f"  jackknife sample {jk}/{n_samples}")
    for r_idx in range(r_max):
        W_rt = wilson_jk[jk, 1:T, r_idx]    # skip t=0
        try:
            p0 = [W_rt[0] if W_rt[0] > 0 else 1.0, 0.3]
            popt, _ = curve_fit(wilson_exp, t_vals, W_rt, p0=p0, maxfev=5000)
            A_jk[jk, r_idx] = popt[0]
            V_jk[jk, r_idx] = popt[1]
        except Exception:
            A_jk[jk, r_idx] = np.nan
            V_jk[jk, r_idx] = np.nan

V_mean, V_err = jackknife_stats(V_jk)
A_mean, A_err = jackknife_stats(A_jk)

print("\nPotential V(r):")
print(f"  {'r':>4}  {'V(r)':>10}  {'err':>10}")
for i in range(r_max):
    print(f"  {r_vals[i]:>4.0f}  {V_mean[i]:>10.6f}  {V_err[i]:>10.6f}")

# --- Cornell fit V(r) = -A_c/r + sigma*r + C ---
print("\nFitting Cornell potential V(r) = -A/r + sigma*r + C ...")

def fit_cornell_jk(V_jk_samples, r, V_err):
    n = V_jk_samples.shape[0]
    params_jk = np.zeros((n, 3))
    for i in range(n):
        try:
            popt, _ = curve_fit(cornell, r, V_jk_samples[i], p0=[0.3, 0.1, 0.0], 
                                sigma=V_err, absolute_sigma=True, maxfev=5000)
            params_jk[i] = popt
        except Exception:
            params_jk[i] = np.nan
    return params_jk

if r_max >= 4:
    cornell_jk = fit_cornell_jk(V_jk, r_vals, V_err)
    cornell_mean, cornell_err = jackknife_stats(cornell_jk)
    A_c_mean, sigma_mean, C_mean = cornell_mean
    A_c_err,  sigma_err,  C_err  = cornell_err
    print(f"  A_coulomb = {A_c_mean:.6f} ± {A_c_err:.6f}")
    print(f"  sigma     = {sigma_mean:.6f} ± {sigma_err:.6f}")
    print(f"  C         = {C_mean:.6f} ± {C_err:.6f}")
    cornell_ok = sigma_mean > 0
else:
    # Too few r points for 3-parameter Cornell fit — use linear V(r) = sigma*r + C
    print(f"  NOTE: only {r_max} r-values available (S={S} too small for full Cornell fit)")
    print(f"  Falling back to linear fit: V(r) = sigma*r + C  (no Coulomb term)")
    def linear_V(r, sigma, C):
        return sigma * r + C
    lin_jk = np.zeros((n_samples, 2))
    for i in range(n_samples):
        try:
            popt, _ = curve_fit(linear_V, r_vals, V_jk[i], p0=[0.1, 0.0],
                                sigma=V_err, absolute_sigma=True)
            lin_jk[i] = popt
        except Exception:
            lin_jk[i] = np.nan
    lin_mean, lin_err = jackknife_stats(lin_jk)
    sigma_mean, C_mean   = lin_mean
    sigma_err,  C_err    = lin_err
    A_c_mean = A_c_err   = 0.0   # no Coulomb term
    print(f"  sigma = {sigma_mean:.6f} ± {sigma_err:.6f}")
    print(f"  C     = {C_mean:.6f} ± {C_err:.6f}")
    cornell_ok = sigma_mean > 0
    # build fake cornell_jk for r0 calculation
    cornell_jk = np.column_stack([np.zeros(n_samples), lin_jk[:, 0], lin_jk[:, 1]])

# --- Sommer parameter r0: r0^2 * (A_c/r0^2 + sigma) = 1.65  →  A_c + sigma*r0^2 = 1.65 ---
def sommer_r0(A_c, sigma):
    val = (1.65 - A_c) / sigma
    if val <= 0:
        return np.nan
    return np.sqrt(val)

r0_jk = np.array([sommer_r0(cornell_jk[i, 0], cornell_jk[i, 1]) for i in range(n_samples)])
if not cornell_ok:
    print("  WARNING: sigma <= 0 from Cornell fit — Cornell/Sommer r0 cannot be extracted on this lattice size.")
r0_mean, r0_err = jackknife_stats(r0_jk)

print(f"\nSommer parameter r0/a = {r0_mean:.4f} ± {r0_err:.4f}  (lattice units)")

# lattice spacing from Wilson method (in fm)
a_wilson    = r0_phys_fm / r0_mean
a_wilson_err = r0_phys_fm * r0_err / r0_mean**2
print(f"  => a (Wilson) = {a_wilson*1000:.2f} ± {a_wilson_err*1000:.2f}  [10^-3 fm]  = {a_wilson:.4f} fm")

# ═══════════════════════════════════════════════════════════════════════════════
# METHOD 2: Gradient flow → t0 scale
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("METHOD 2: Gradient Flow → t0 Scale")
print("="*60)

# energy_data[s, config]: E(t_s) per config at smear step s
# flow_times[s]: flow time t_s = s * flow_steps_per_group * epsilon

# Apply correct normalization: E_correct = beta * E_raw
# where E_raw = 6 - calculate_plaquette()  (dimensionless, in lattice units)
# Derivation: 1 - (1/3)ReTrP ≈ (g0^2/12)*a^4 * Σ_a F_μν^a F_μν^a (leading order)
# Luscher energy density E(t) = (1/2V)Σ_{x,μ<ν,a} (F_μν^a)^2 = β * E_raw
# (using β = 6/g0^2, and the factor 2β cancels the 1/2 from the energy density definition)
# Reference: Luscher & Weisz arXiv:1006.4518, t0 defined by t^2 E(t0) = 0.3
# NOTE: requires L >> sqrt(8*t0) to avoid finite-size effects
#   At beta=6.5, t0/a^2 ~ 5-6, sqrt(8*5.5) ~ 6.6a  => need L >= 13a for good extraction

# Compute t^2 * E(t) for each smear step, averaged with jackknife
t2E_jk_all = np.zeros((n_samples, smear_steps))
norm = beta   # correct: E(t) = β * E_raw
E_s=np.zeros((n_samples, smear_steps))
for s in range(smear_steps):
    E_s_raw = energy_data[s]                        # E_raw = 6 - P_raw,  [configs]
    t_s = flow_times[s]
    E_s_jk = jackknife_1d(E_s_raw) * norm           # apply normalization
    t2E_jk_all[:, s] = t_s**2 * E_s_jk
    E_s[:,s]=E_s_jk

# flow radius at each step: r_flow = sqrt(8*t)
r_flow = np.sqrt(8 * flow_times)
finite_size_limit = S / 2.0    # flows beyond r_flow ~ L/2 are contaminated

t2E_mean, t2E_err = jackknife_stats(t2E_jk_all)
E_mean,E_err=jackknife_stats(E_s)
print("t^2 * <E(t)> at each flow step:")
for s in range(smear_steps):
    print(f"  t = {flow_times[s]:.4f}  =>  t^2*E = {t2E_mean[s]:.6f} ± {t2E_err[s]:.6f}")


# --- Find t0: interpolate where t^2*<E(t)> = 0.3 ---
target = 0.3

def find_t0(flow_t, t2E_vals):
    """Interpolate to find t0 where t2E = 0.3."""
    # find bracketing interval
    for i in range(len(t2E_vals) - 1):
        if (t2E_vals[i] - target) * (t2E_vals[i+1] - target) < 0:
            # linear interpolation
            t_lo, t_hi = flow_t[i], flow_t[i+1]
            v_lo, v_hi = t2E_vals[i], t2E_vals[i+1]
            t0 = t_lo + (target - v_lo) * (t_hi - t_lo) / (v_hi - v_lo)
            return t0
    return np.nan

t0_jk = np.array([
    find_t0(flow_times, t2E_jk_all[i])
    for i in range(n_samples)
])

t0_mean, t0_err = jackknife_stats(t0_jk)

if np.isnan(t0_mean):
    print(f"\nFlow scale: t0 not found in data range (0.3 > max t²E = {t2E_mean[-1]:.4f})")
else:
    print(f"\nFlow scale t0/a^2 = {t0_mean:.6f} ± {t0_err:.6f}  (lattice units)")

# Calculate r0 from gradient flow: r0/a = sqrt(8*t0/a^2) * (0.5/0.415)
# Physical: sqrt(8*t0) ~ 0.415 fm, r0 ~ 0.5 fm
if not np.isnan(t0_mean):
    sqrt_t0_mean = np.sqrt(t0_mean)
    sqrt_t0_err  = 0.5 * t0_err / sqrt_t0_mean
    print(f"  sqrt(t0)/a = {sqrt_t0_mean:.4f} ± {sqrt_t0_err:.4f}  (lattice units)")
    
    # r0 from gradient flow
    r0_flow_phys = 0.5 / sqrt8t0_phys_fm  # = 0.5/0.415 = 1.2048
    r0_from_flow = np.sqrt(8.0 * t0_mean) * r0_flow_phys
    r0_from_flow_err = np.sqrt(8.0) * r0_flow_phys * (0.5 * t0_err / sqrt_t0_mean)
    print(f"  r0/a (from flow) = {r0_from_flow:.4f} ± {r0_from_flow_err:.4f}  (lattice units)")
    
    # lattice spacing from gradient flow method (in fm)
    # sqrt(8*t0) ~ 0.415 fm  =>  sqrt(8) * sqrt(t0)/a * a = 0.415 fm
    # a = 0.415 / (sqrt(8) * sqrt(t0)/a)  fm
    a_flow     = sqrt8t0_phys_fm / (np.sqrt(8) * sqrt_t0_mean)
    a_flow_err = sqrt8t0_phys_fm * sqrt_t0_err / (np.sqrt(8) * sqrt_t0_mean**2)
    print(f"  => a (flow) = {a_flow*1000:.2f} ± {a_flow_err*1000:.2f}  [10^-3 fm]  = {a_flow:.4f} fm")
else:
    print("  WARNING: t0 not found in measured flow time range — increase smear_steps or flow_steps_per_group")
    a_flow = a_flow_err = np.nan
    r0_from_flow = r0_from_flow_err = np.nan

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating plots...")

# ── Plot 1: Wilson loop fits W(r,t) vs t ──────────────────────────────────────
n_r_plot = min(r_max, 4)
fig, axes = plt.subplots(1, n_r_plot, figsize=(4 * n_r_plot, 4), sharey=False)
if n_r_plot == 1:
    axes = [axes]

W_mean_all = np.mean(wilson_jk, axis=0)    # [T, S]
W_err_all  = np.std(wilson_jk, axis=0) * np.sqrt(n_samples - 1)

for r_idx in range(n_r_plot):
    ax = axes[r_idx]
    W_m = W_mean_all[1:T, r_idx]
    W_e = W_err_all[1:T, r_idx]
    ax.errorbar(t_vals, W_m, W_e, fmt='o', capsize=3, label='Data')
    t_fit = np.linspace(t_vals[0], t_vals[-1], 100)
    ax.plot(t_fit, wilson_exp(t_fit, A_mean[r_idx], V_mean[r_idx]), '-',
            label=f'V={V_mean[r_idx]:.4f}')
    ax.set_xlabel('t/a')
    ax.set_ylabel('W(r,t)')
    ax.set_title(f'r = {r_idx+1}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Wilson Loops  (T={T}, S={S}, β={beta})', fontsize=13)
plt.tight_layout()
out1 = os.path.join(output_dir, f'wilson_loop_fit_T{T}_S{S}_beta{beta:.1f}.png')
plt.savefig(out1, dpi=300)
plt.close()
print(f"  Saved: {out1}")

# ── Plot 2: Cornell potential V(r) ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(r_vals, V_mean, V_err, fmt='o', capsize=4, markersize=7,
            color='royalblue', label='V(r) from Wilson loops')

r_cont = np.linspace(r_vals[0] * 0.9, r_vals[-1] * 1.1, 200)
if r_max >= 3:
    ax.plot(r_cont, cornell(r_cont, A_c_mean, sigma_mean, C_mean),
            '-', color='tomato', label=f'Cornell fit\nσ={sigma_mean:.4f}, A={A_c_mean:.4f}')
else:
    ax.plot(r_cont, sigma_mean * r_cont + C_mean,
            '-', color='tomato', label=f'Linear fit (S too small)\nσ={sigma_mean:.4f}')

if not np.isnan(r0_mean):
    ax.axvline(r0_mean, color='green', linestyle='--', alpha=0.7,
               label=f'r₀/a = {r0_mean:.3f} ± {r0_err:.3f}')

ax.set_xlabel('r/a', fontsize=12)
ax.set_ylabel('V(r)', fontsize=12)
ax.set_title(f'Cornell Potential  (T={T}, S={S}, β={beta})', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
out2 = os.path.join(output_dir, f'cornell_potential_T{T}_S{S}_beta{beta:.1f}.png')
plt.savefig(out2, dpi=300)
plt.close()
print(f"  Saved: {out2}")

# ── Plot 3: Gradient flow t²E(t) ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

ax.errorbar(flow_times, t2E_mean, t2E_err, fmt='o-', capsize=4, markersize=7,
            color='darkorange', label='t²⟨E(t)⟩')
ax.axhline(0.3, color='gray', linestyle='--', alpha=0.7, label='t₀ target = 0.3')


if not np.isnan(t0_mean):
    ax.axvline(t0_mean, color='purple', linestyle=':', alpha=0.8,
               label=f't₀/a² = {t0_mean:.4f}')
else:
    ax.text(0.5, 0.6, f't₀ not reached\n(0.3 not in data range\nfor β={beta})',
            transform=ax.transAxes, ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax.set_xlabel('t / a²', fontsize=12)
ax.set_ylabel('t² ⟨E(t)⟩', fontsize=12)
ax.set_title(f'Gradient Flow  (T={T}, S={S}, β={beta})', fontsize=13)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out3 = os.path.join(output_dir, f'gradient_flow_scale_T{T}_S{S}_beta{beta:.1f}.png')
plt.savefig(out3, dpi=300)
plt.close()
print(f"  Saved: {out3}")

# ── Plot 4: Scale comparison ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))

methods = ['Wilson loop\n(Sommer r₀)', 'Gradient flow\n(t₀ scale)']
a_vals  = np.array([a_wilson,     a_flow])     * 1000   # convert to 10^-3 fm
a_errs  = np.array([a_wilson_err, a_flow_err]) * 1000
colors  = ['royalblue', 'darkorange']

for i, (m, av, ae, c) in enumerate(zip(methods, a_vals, a_errs, colors)):
    ax.errorbar(i, av, ae, fmt='o', capsize=6, markersize=10,
                color=c, label=f'{m}\na = {av:.2f} ± {ae:.2f} ×10⁻³ fm')

ax.set_xticks([0, 1])
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylabel('a  [×10⁻³ fm]', fontsize=12)
ax.set_title(f'Scale Setting Comparison  (β={beta})', fontsize=13)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
out4 = os.path.join(output_dir, f'scale_comparison_T{T}_S{S}_beta{beta:.1f}.png')
plt.savefig(out4, dpi=300)
plt.close()
print(f"  Saved: {out4}")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SCALE SETTING SUMMARY")
print("="*60)
print(f"  Lattice:  T={T}, S={S}, β={beta}")
print(f"  Configs:  {n_samples}")
print()
print(f"  Method 1 (Wilson / Sommer):")
print(f"    r0/a   = {r0_mean:.4f} ± {r0_err:.4f}")
print(f"    a      = {a_wilson*1000:.3f} ± {a_wilson_err*1000:.3f}  [10^-3 fm]")
print()
print(f"  Method 2 (Gradient flow / t0):")
if not np.isnan(t0_mean):
    print(f"    t0/a^2 = {t0_mean:.4f} ± {t0_err:.4f}")
    print(f"    sqrt(t0)/a = {sqrt_t0_mean:.4f} ± {sqrt_t0_err:.4f}")
    print(f"    a      = {a_flow*1000:.3f} ± {a_flow_err*1000:.3f}  [10^-3 fm]")
    diff = abs(a_wilson - a_flow)
    combined_err = np.sqrt(a_wilson_err**2 + a_flow_err**2)
    print()
    print(f"  Difference: {diff*1000:.3f} × 10^-3 fm  ({diff/combined_err:.1f} sigma)")
else:
    print("    t0 not found — need more flow steps")
print("="*60)
