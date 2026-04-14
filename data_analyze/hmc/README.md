# HMC Lattice QCD Analysis Scripts

Analysis suite for SU(3) pure gauge HMC simulation data. All scripts auto-discover datasets under `../../hmc/results/{lattice}/{beta}/flow/`.

## Dependencies

numpy, pandas, matplotlib, scipy (available in the project venv).

## Data Format

Each dataset directory (`results/{lattice}/{beta}/flow/`) contains:

| File | Columns | Description |
|------|---------|-------------|
| `Plaq.csv` | `traj, plaq` | Average plaquette per trajectory |
| `smear.csv` | `traj, t_flow, Q, E` | Gradient flow observables (topological charge Q, energy density E) at each flow time |
| `WS_loop.csv` | `traj, r, t, W` | Smeared Wilson loops W(r,t) for static potential extraction |

## Scripts

### analyze_flow_topo.py

Comprehensive HMC diagnostics: plaquette history, topological charge history/histogram, gradient flow evolution of E(t) and Q(t), scale setting via t^2 E(t) = 0.3 (t0 extraction), and autocorrelation analysis of Q.

**Outputs:** `plaquette_history.png`, `topo_charge.png`, `flow_evolution.png`, `energy_t2E.png`, `autocorr_Q.png`, `topo_vs_flowtime.png`

### analyze_scale_t0.py

Extracts the gradient flow scale t0/a^2 from the condition t^2 <E(t)> = 0.3 with jackknife error estimation. Computes the lattice spacing a in fm using sqrt(8 t0)|phys = 0.415 fm.

**Outputs:** `scale_t0.png`

### analyze_sommer.py

Sommer parameter r0 from the static quark potential V(r). Fits W(r,t) = A exp(-V t) with correlated chi^2 using jackknife covariance, then fits the Cornell potential V(r) = V0 + sigma*r - e/r. Applies the Sommer condition r0^2 dV/dr|r0 = 1.65 to extract r0/a and the lattice spacing.

**Outputs:** `sommer_wloop_fit.png`, `sommer_potential.png`, `sommer_plateau.png`

### analyze_topo_distribution.py

Topological charge Q distribution as a function of gradient flow time. Plots Q histograms at selected flow times and tracks mean/std of Q vs flow time.

**Outputs:** `topo_distribution.png`, `topo_std_vs_flow.png`

### analyze_wilson_loop.py

Diagnostic plots for Wilson loop W(r,t) data: W vs t for each r (with jackknife errors), W vs r for selected t, log-scale heatmap, and effective potential V_eff(r,t) = -ln(W(r,t)/W(r,t+1)).

**Outputs:** `wilson_loop_vs_t.png`, `wilson_loop_vs_r.png`, `wilson_loop_heatmap.png`, `wilson_loop_Veff.png`

## Output

All plots are saved to `plots/{lattice}/{beta}/`.

## Usage

```bash
source /home/khw/Documents/Git_repository/qcd/data_analyze/.venv/bin/activate
python analyze_flow_topo.py
python analyze_scale_t0.py
python analyze_sommer.py
python analyze_topo_distribution.py
python analyze_wilson_loop.py
```
