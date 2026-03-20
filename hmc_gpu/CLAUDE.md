# CLAUDE.md — Project Instructions for Claude Code

## Project Overview
This is a CUDA/C++ GPU-accelerated Hybrid Monte Carlo (HMC) simulation for lattice QCD.
The code simulates SU(3) gauge fields on a 4D lattice using GPU kernels.

## Build System
- Uses **CMake** with presets defined in `CMakePresets.json`
- Build directory: `build/`
- To configure and build:
  ```bash
  cmake --preset default
  cmake --build build/
  ```
- Requires CUDA toolkit and a GPU with compute capability >= 7.0

## Project Structure
```
include/
  structs/     # Core data types: complex.cuh, matrix.cuh, lattice.cuh, spinor.cuh
  linalg/      # Linear algebra utilities
  physics/     # Physics-specific headers
  utils/       # General utility headers
lib/kernels/   # CUDA kernel implementations (.cu files)
tests/         # Test and benchmark executables
```

## Code Conventions
- All GPU kernels are in `lib/kernels/` as `.cu` files
- Headers use `.cuh` for CUDA headers, `.h` for pure C++ headers
- Matrix type is SU(3): 3x3 complex matrices
- Lattice indices follow row-major ordering: `[t][z][y][x]`
- Use `__device__` functions for GPU-only utilities

## Key Types (in `include/structs/`)
- `Complex` — complex number type (`complex.cuh`)
- `Matrix` — SU(3) 3x3 matrix (`matrix.cuh`)
- `Lattice` — gauge field on 4D lattice (`lattice.cuh`)
- `Spinor` — fermion spinor field (`spinor.cuh`)

## Workflow Preferences
- Always read relevant headers before modifying kernel code
- When adding a new kernel, follow the pattern in existing `.cu` files
- Do not auto-commit — always ask before committing
- Prefer editing existing files over creating new ones

## Testing
- Test files are in `tests/` and are standalone CUDA executables
- Run individual tests by building the specific target, e.g.:
  ```bash
  cmake --build build/ --target test_matrix_def
  ./build/tests/test_matrix_def
  ```

---

## Active Research Task: Scale Setting via Two Methods

**Goal**: Verify that two independent methods of setting the lattice scale `a` agree with each other.

### Physics Background

The lattice spacing `a` is not an input — it must be extracted from measurable dimensionless quantities and matched to known physical values. Two standard methods exist:

**Method 1 — Static Quark Potential (Wilson loops)**
- Compute W(r, t) = `⟨(1/3) Re Tr[loop of size r×t]⟩` for all spatial separations r and temporal extents t
- Fit each W(r, t) vs t to extract the static potential: `W(r, t) = A(r) * exp(-V(r) * t)`
- Fit V(r) to the Cornell potential: `V(r) = -A/r + σ·r + C`
  - σ = string tension, A = Coulomb coefficient, C = constant
- Extract the Sommer parameter `r₀/a` in lattice units, defined by:
  `r₀² · F(r₀) = 1.65` where `F(r) = dV/dr = A/r² + σ`
  → Solving: `r₀ = sqrt((1.65 - A) / σ)` (in lattice units)
- Physical value: `r₀ ≈ 0.5 fm`

**Method 2 — Yang-Mills Gradient Flow (t₀ scale)**
- Under gradient flow at flow time t, compute the energy density:
  `E(t) = (1/V) * Σ_{x,μ<ν} [1 - (1/3) Re Tr[P_μν(x)]]`
  In terms of existing observable: `E(t) = 6.0 - calculate_plaquette()`
- Find flow time `t₀` defined by: `t₀² · ⟨E(t₀)⟩ = 0.3`
- Physical value: `sqrt(8 t₀) ≈ 0.415 fm`

**Consistency check**: Both methods measure the same physical scale:
`a = r₀_lattice * (0.5 fm / r₀_phys) = sqrt(t₀_lattice) * (0.415 / sqrt(8)) fm`
These two estimates of `a` should agree.

---

### Implementation Plan

#### Step 1 — Re-enable Wilson loop measurements in `tests/autocorrelate_hmc.cu`
Current state: Wilson loop measurement block is commented out (lines 87–99, 168–179).
Action:
- Uncomment the Wilson loop measurement block inside the production loop (at `s == 0`, before gradient flow)
- Uncomment the Wilson loop file-saving block
- Ensure `h_wilson_data` is properly filled: index = `traj * n_wilson_loops + (t_ext-1)*max_x + (x_ext-1)`
- Loops are over spatial direction `mu=0` (x) and temporal `nu=3` (t), sizes from 1 to max_x and 1 to max_t

#### Step 2 — Add energy density E(t) output per gradient flow step in `tests/autocorrelate_hmc.cu`
Current state: `h_plaq_data` stores plaquette at each flow step, but flow time `t` is not explicitly recorded.
Action:
- Add storage array `h_energy_data[n_samples * smear_steps]`
- At each flow step `s` (after applying gradient flow), record:
  - Cumulative flow time: `t_flow = (s + 1) * flow_steps_per_group * epsilon`
  - Energy density: `E = 6.0 - calculate_plaquette()` (note: plaquette already calculated)
  - Dimensionless observable: `t² * E`
- Add `flow_times[smear_steps]` array to header of output file so Python knows t values
- Save to file: `energy_density_data.txt` with header line containing flow times

#### Step 3 — (Optional CUDA) Add signed topological charge output
Current state: `topo_charge()` returns `Q²` (line 163 of `phy_obse.cu`).
If needed for analysis, add a separate `topo_charge_signed()` method that returns `Q` not `Q²`.
This is optional — not required for scale setting.

#### Step 4 — Python analysis script
**File**: `/home/khw/Documents/Git_repository/qcd/data_analyze/scale_setting.py`

Script structure:
1. **Load Wilson loop data** from `wilson_loop_data.txt`, reshape to `[configs, t_ext, r_ext]`
2. **Jackknife resampling** on Wilson loop data (reuse pattern from `wilson_loop_analysis.py`)
3. **Fit W(r,t) → V(r)**:
   - For each r, fit W vs t to `A * exp(-V*t)` using t ∈ [1, T-2] (avoid boundaries)
   - Use full jackknife: fit each jackknife sample, get V(r) ± error
4. **Fit Cornell potential V(r)**:
   - Fit `V(r) = -A/r + σ*r + C` using nonlinear least squares
   - Extract σ (string tension) and A (Coulomb coefficient) with jackknife errors
5. **Compute Sommer parameter** `r₀/a`:
   - Solve `r₀² * (A/r₀² + σ) = 1.65` → `r₀ = sqrt((1.65 - A) / σ)` (in lattice units)
6. **Load energy density data** from `energy_density_data.txt`
7. **Compute t² * E(t)** vs flow time t at each flow step (averaged over configs with jackknife error)
8. **Find t₀**: interpolate where `t² * ⟨E(t)⟩ = 0.3` → gives `sqrt(t₀)/a` in lattice units
9. **Compare lattice spacings**:
   - From Wilson: `a_W = 0.5 fm / (r₀/a)`
   - From flow: `a_F = (0.415/sqrt(8)) fm / (sqrt(t₀)/a)`
   - Plot both on the same figure with error bars, check consistency
10. **Plots to produce**:
    - `wilson_loop_fit.png` — W(r,t) data + exponential fits for each r
    - `cornell_potential.png` — V(r) data + Cornell fit, Sommer point r₀ marked
    - `gradient_flow_scale.png` — t²E(t) vs t, horizontal line at 0.3, t₀ marked
    - `scale_comparison.png` — both estimates of `a` side by side

---

### Data File Format (output of C++ simulation)

All files saved to: `hmc_results/t{T}_s{S}_beta{β}/`

| File | Format | Description |
|---|---|---|
| `topo_charge_data.txt` | `smear_steps * n_samples` rows | Q² per config per flow group |
| `plaquette_data.txt` | `smear_steps * n_samples` rows | P per config per flow group |
| `wilson_loop_data.txt` | `T * S * n_samples` rows | W(r,t) for all r,t (at flow step 0 only) |
| `energy_density_data.txt` | header + `smear_steps * n_samples` rows | E(t) per config per flow group |

### Analysis File Location

All Python analysis scripts go in:
`/home/khw/Documents/Git_repository/qcd/data_analyze/`

New file to create: `scale_setting.py`
Existing files to reference/reuse: `wilson_loop_analysis.py` (jackknife pattern)
