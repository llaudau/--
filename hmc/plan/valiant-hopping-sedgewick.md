# Plan: Wilson Flow Observables + Data Analysis Overhaul

## Context
The current production loop mixes APE and flow modes and only records plaquette + Q.
The goal is to:
- Drop APE from the *run loop* (keep the function), making flow the sole smearing path
- Per trajectory: record raw plaquette → `Plaq.csv`; record Wilson loops → `WS_loop.csv`
- Per flow step: record (t_flow, Q, E) → `smear.csv`
- Add three Python analysis scripts for (1) Q distribution vs flow time, (2) t₀ scale setting, (3) Sommer parameter r₀

---

## Files to modify

### C++ simulation

| File | Change |
|---|---|
| `include/params.hpp` | Change default `smear_method` to `"flow"`; add `wl_r_max=6`, `wl_t_max=8`; add CLI flags `--wl_r` and `--wl_t` |
| `include/gradient_flow.hpp` | Change callback signature: `(double t, double Q, double E, void* ctx)` — drop plaq, add E |
| `src/gradient_flow.cpp` | Call `clover_energy(gf)` inside `run_flow` at each step; pass to callback |
| `apps/run.cpp` | Rewrite production loop (see detail below) |

### Python analysis (new files)

| File | Purpose |
|---|---|
| `data_analyze/hmc/analyze_topo_distribution.py` | Q histograms at each flow time step |
| `data_analyze/hmc/analyze_scale_t0.py` | t²E(t) → t₀ extraction with jackknife |
| `data_analyze/hmc/analyze_sommer.py` | V(r) from W(r,t) → Sommer parameter r₀ |

---

## Detailed changes

### 1. `include/params.hpp`

```cpp
std::string smear_method = "flow";   // was "ape"
```

No new Wilson loop parameters needed — `wl_r_max` defaults to `Lx` and `wl_t_max` defaults to `Lt`, both already available in `SimParams`. Add to `print()`:
```cpp
std::cout << "Wilson loop: r_max=" << Lx << " (=Ls), t_max=" << Lt << "\n";
```

---

### 2. `include/gradient_flow.hpp`

Change callback typedef:
```cpp
// old: void (*callback)(double t, double plaq, double Q, void* ctx)
// new:
void run_flow(const GaugeField& gf_orig, double eps, double t_max,
              void (*callback)(double t, double Q, double E, void* ctx), void* ctx);
```

---

### 3. `src/gradient_flow.cpp`

Add `clover_energy` call inside `run_flow`:
```cpp
for(int i=0; i<n_steps; i++) {
    flow_step(gf, eps);
    double t = (i+1) * eps;
    gf.exchange_halo();
    double Q = topo_charge(gf);
    double E = clover_energy(gf);           // NEW
    if(callback) callback(t, Q, E, ctx);   // drop plaq, add E
}
```

---

### 4. `apps/run.cpp`

**New output files:** `Plaq.csv`, `WS_loop.csv`, `smear.csv`

**New FlowCtx struct:**
```cpp
struct FlowCtx {
    std::vector<double> t_vals, Q_vals, E_vals;
};
void flow_cb(double t, double Q, double E, void* ctx) {
    FlowCtx* fc = (FlowCtx*)ctx;
    fc->t_vals.push_back(t);
    fc->Q_vals.push_back(Q);
    fc->E_vals.push_back(E);
}
```

**Production loop (replacing current APE/flow if-else):**
```
for each trajectory n:
    1. hmc_step(gf)
    2. gf.exchange_halo()

    // --- Plaq.csv ---
    3. plaq = plaquette(gf)
       append_csv(plaq_path, "traj,plaq", {n, plaq})

    // --- WS_loop.csv (on original config, before flow) ---
    4. for r = 1..p.Lx:         // spatial extent = Ls
           for t = 1..p.Lt:     // temporal extent
               W = average of wilson_loop(gf, r, t, mu, 3) over mu=0,1,2
               append_csv(wl_path, "traj,r,t,W", {n, r, t, W})

    // --- smear.csv (Wilson flow on copy) ---
    5. FlowCtx fc
       run_flow(gf, flow_eps, flow_t_max, flow_cb, &fc)
       for each flow step i:
           append_csv(smear_path, "traj,t_flow,Q,E",
                      {n, fc.t_vals[i], fc.Q_vals[i], fc.E_vals[i]})
```

**Remove:** entire APE branch (`if(p.smear_method == "ape") { ... }`).
Wilson loop measurement is on the **original (unsmeared)** config before flow.

**CSV paths:**
```cpp
std::string plaq_path  = p.output_dir + "/Plaq.csv";
std::string wl_path    = p.output_dir + "/WS_loop.csv";
std::string smear_path = p.output_dir + "/smear.csv";
```

---

### 5. `data_analyze/hmc/analyze_topo_distribution.py`

**Purpose:** Show how Q sharpens toward integers as flow time increases.

**Inputs:** `smear.csv` (`traj, t_flow, Q, E`)

**Output plots:**
- Grid of Q histograms at selected t_flow values (e.g. t=0.1, 0.3, 0.5, 1.0, 2.0)
- Mean and std of Q vs t_flow (convergence plot)

---

### 6. `data_analyze/hmc/analyze_scale_t0.py`

**Purpose:** Extract t₀ from t²⟨E(t)⟩ = 0.3.

**Inputs:** `smear.csv` (`traj, t_flow, Q, E`)

**Algorithm:**
1. Average E over trajectories at each t_flow → ⟨E(t)⟩
2. Compute t²⟨E(t)⟩
3. Find t₀ by linear interpolation where t²E = 0.3
4. Jackknife error on t₀
5. Convert: a [fm] = 0.1465 fm / √(t₀/a²)   (using √8t₀|_phys = 0.415 fm)

**Output plots:**
- t²⟨E(t)⟩ vs t with horizontal line at 0.3, vertical at t₀
- E(t) per trajectory (few samples, to show noise)

---

### 7. `data_analyze/hmc/analyze_sommer.py`

**Purpose:** Extract Sommer parameter r₀ from the static quark potential.

**Inputs:** `WS_loop.csv` (`traj, r, t, W`)

**Algorithm:**
1. Average W(r,t) over trajectories
2. Form effective potential: V_eff(r,t) = ln[⟨W(r,t)⟩ / ⟨W(r,t+1)⟩]
3. Identify plateau in t for each r → V(r)
4. Fit Cornell potential: V(r) = V₀ + σr − e/r
5. Sommer condition: r₀² dV/dr|_{r₀} = 1.65 → r₀ = √((1.65 − e)/σ)
6. Jackknife errors on V(r) and on r₀

**Output plots:**
- V_eff(r,t) vs t for each r (plateau check)
- V(r) with Cornell fit overlaid
- r₀/a value with error bar

---

## CSV schema summary

| File | Columns | Written when |
|---|---|---|
| `Plaq.csv` | `traj, plaq` | Once per trajectory (unsmeared) |
| `WS_loop.csv` | `traj, r, t, W` | Once per trajectory × r × t (unsmeared) |
| `smear.csv` | `traj, t_flow, Q, E` | Once per flow step × trajectory |

---

## Verification
1. Build: `make` or `cmake --build build`
2. Short test run: `mpirun -np 1 ./build/hmc_run --Lx 4 --Lt 8 --thermal 5 --samples 3 --flow_t 0.5 --wl_r 3 --wl_t 3`
3. Check all three CSVs exist with correct headers and row counts
4. Run each analysis script and verify plots are generated without errors
