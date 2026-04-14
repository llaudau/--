# HMC SU(3) Lattice QCD — CPU / MPI+OpenMP

A Hybrid Monte Carlo (HMC) simulation for SU(3) pure gauge theory on a 4D lattice, parallelized with MPI (domain decomposition along the **z-axis**) and OpenMP (shared-memory threading within each rank). Designed to run on the **Tianhe** supercomputer (64 cores/node, 8 NUMA domains).

## File Structure

```
hmc/
├── CMakeLists.txt          # CMake build (C++17, MPI, OpenMP, -O3)
├── Makefile                # Top-level convenience wrapper (Tianhe mpicxx path)
├── auto_run.sh             # Auto-parallel SLURM launcher (computes optimal layout)
│
├── apps/
│   └── run.cpp             # Main entry point (thermalization + production loop)
│
├── include/
│   ├── params.hpp          # SimParams struct + CLI parser (--key value)
│   ├── lattice.hpp         # GaugeField, LatticeLayout, MPI halo exchange
│   ├── matrix.hpp          # Fixed-size N×N complex matrix (SU(3))
│   ├── complex.hpp         # Complex number type
│   ├── gauge_ops.hpp       # Gauge link operations (staples, etc.)
│   ├── hmc.hpp             # HMC molecular dynamics step
│   ├── observables.hpp     # Plaquette, topological charge
│   ├── gradient_flow.hpp   # Wilson gradient flow
│   ├── ape.hpp             # APE smearing
│   └── io.hpp              # CSV output, directory helpers
│
├── src/
│   ├── lattice.cpp         # GaugeField implementation
│   ├── hmc.cpp             # HMC step (leapfrog integrator)
│   ├── observables.cpp     # Plaquette & topological charge measurement
│   ├── gradient_flow.cpp   # Wilson flow implementation
│   ├── ape.cpp             # APE smearing implementation
│   └── io.cpp              # File I/O utilities
│
├── build/                  # Build output (cmake artifacts, hmc_run binary)
├── results/                # Simulation output (auto-organized by lattice/beta/smear)
└── plan/                   # Planning notes
```

## Build

**Prerequisites:** C++17 compiler, MPI, OpenMP.

### Using the top-level Makefile (Tianhe)

```bash
make            # configure + build (16 parallel jobs)
make clean      # remove build/
make rebuild    # clean + build
```

The Makefile hardcodes the Tianhe MPI compiler path (`/usr/local/ompi/bin/mpicxx`).

### Using CMake directly

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The resulting binary is `build/hmc_run`.

## Command-Line Arguments

All arguments use `--key value` format. Unspecified parameters use defaults.

| Flag | Default | Description |
|------|---------|-------------|
| `--Lx` | 16 | Spatial lattice size X |
| `--Ly` | 16 | Spatial lattice size Y |
| `--Lz` | 16 | Spatial lattice size Z (decomposed axis, must be multiple of 8×nodes) |
| `--Lt` | 32 | Temporal lattice size |
| `--beta` | 6.0 | Coupling constant |
| `--traj` | 0.2 | HMC trajectory length |
| `--md` | 40 | Leapfrog MD integration steps |
| `--thermal` | 1000 | Thermalization sweeps |
| `--samples` | 1000 | Production measurements |
| `--print` | 20 | Print interval (every N steps) |
| `--skip` | 0 | HMC updates between measurements (0 = measure every trajectory) |
| `--flow_eps` | 0.04 | Wilson flow step size epsilon |
| `--flow_t` | 2.0 | Wilson flow max flow time |
| `--omp` | 8 | OpenMP threads per rank (1 per NUMA domain) |
| `--seed` | 42 | RNG seed |
| `--out` | (auto) | Output directory (default: `results/T{Lt}_S{Lx}/beta{beta}/flow`) |

### Example

```bash
# Local test: 2 MPI ranks splitting Lz=4
mpirun -np 2 ./build/hmc_run --Ls 4 --Lt 8 --beta 6.0 --thermal 5 --samples 3 --flow_t 0.5 --omp 4
```

### Typical production run

```bash
# 16×8³, β=6.0, 5000 configs, traj length 1.0, 20 MD steps, 10000 thermals, 5 skip
# On Tianhe (1 node, 8 MPI ranks):
./auto_run.sh --nodes 1 --Ls 8 --Lt 16 --beta 6.0 --traj 1.0 --md 20 --thermal 10000 --samples 5000 --skip 5 --print 100

# Locally (single rank):
mpirun -np 1 ./build/hmc_run --Ls 8 --Lt 16 --beta 6.0 --traj 1.0 --md 20 --thermal 10000 --samples 5000 --skip 5 --print 100 --omp 4
```

With `--skip 5`, each measurement is separated by 6 HMC trajectories (5 skipped + 1 measured), giving 5000 × 6 = 30000 total trajectories.

## Syncing to / from Tianhe

A `.rsyncignore` file excludes build artifacts and large directories. **Always pass `--exclude-from`** or the file is silently ignored.

**Local → Tianhe** (push source code):

```bash
cd /path/to/hmc
rsync -avz --exclude-from=.rsyncignore . tianhe:~/wkh/hmc/
```

**Tianhe → Local** (pull results):

```bash
rsync -avz tianhe:~/wkh/hmc/results/ /path/to/hmc/results/
```

After syncing source code, rebuild on Tianhe (ARM64 — binaries compiled locally will not run):

```bash
ssh tianhe "cd ~/wkh/hmc && make clean && make"
```

---

## Running on Tianhe (SLURM)

### Auto-launcher (recommended)

`auto_run.sh` takes `--nodes` and computes the full SLURM layout automatically:

```bash
# Default lattice 16³×32, β=6.0 on 2 nodes
./auto_run.sh --nodes 2

# Larger lattice
./auto_run.sh --nodes 4 --Lz 32 --Lt 64 --beta 6.2
```

**Constraint:** `Lz` must be divisible by `nodes × 8` (8 ranks per node, 1 per NUMA domain).

| `--nodes` | Ranks | Min Lz |
|-----------|-------|--------|
| 1 | 8 | 8 |
| 2 | 16 | 16 |
| 4 | 32 | 32 |

### Manual SLURM submission

```bash
#SBATCH --partition=thcp1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores

srun ./build/hmc_run --Lx 16 --Ly 16 --Lz 16 --Lt 32 --beta 6.0 --omp 8
```

## Parallelization

**Two-level hierarchy:**

```
Node  (1 physical machine, 64 cores)
├── Rank 0  → NUMA domain 0 (cores  0-7,  8 OMP threads) → Lz_local z-slices
├── Rank 1  → NUMA domain 1 (cores  8-15, 8 OMP threads)
├── ...
└── Rank 7  → NUMA domain 7 (cores 56-63, 8 OMP threads)
```

- **MPI:** Domain decomposition along the **z-axis**. Each rank owns `Lz_local = Lz / (nodes×8)` z-slices plus the full x, y, t extents. Ghost exchange of one z-layer (all x, y, t) between z-neighboring ranks.
- **OpenMP:** 8 threads per rank, pinned to one NUMA domain (`OMP_PROC_BIND=close`). Parallelizes all site loops within the rank.
- **`Lz` constraint:** must be divisible by `nodes × 8`.

### Why z-decomposition?

Decomposing along z keeps t, x, y entirely local to each rank:

- **Wilson loops** in the (x, t) and (y, t) planes are fully local — no MPI boundary crossed regardless of `r` or `t_loop`
- **Wilson loops** in the (z, t) plane work up to `r = Lz_local`
- **Plaquette, topological charge, gradient flow** — all single-hop stencils, unaffected
- t-shifts within a ghost layer are pure index arithmetic (no extra communication)

## Observables (`include/observables.hpp`)

### Plaquette

```
P = (1 / 6 Nc V) Σ_{s, μ<ν} Re Tr[ U_μ(s) U_ν(s+μ̂) U_μ†(s+ν̂) U_ν†(s) ]
```

`Nc = 3`, `V = Lx·Ly·Lz·Lt`. Approaches 1 in the weak-coupling limit. The primary thermalization diagnostic.

---

### Topological Charge — clover-leaf definition

Build the 4-leaf clover average for each plane:

```
C_{μν}(s) = (1/4) Σ_{4 oriented leaves} U_{μν-plaquette}(s)
F_{μν}(s) = ( C_{μν}(s) − C†_{μν}(s) ) / 2        [anti-Hermitian]
```

Plane index: `[0]=F₀₁, [1]=F₀₂, [2]=F₀₃, [3]=F₁₂, [4]=F₁₃, [5]=F₂₃`

```
Q = − 1/(2π²) Σ_s [ −Tr(F₀₂ F₁₃) + Tr(F₀₃ F₁₂) + Tr(F₂₃ F₀₁) ]
```

This is the lattice discretisation of `Q = 1/(32π²) ∫ ε_{μνρσ} Tr[F_{μν} F_{ρσ}] d⁴x`. Integer-valued for smooth configurations.

> **Convention note:** Matches Qlattice `clf_topology_density` ([qcd.cpp](https://github.com/jinluchang/Qlattice/blob/master/qlat/qlat/lib/qcd.cpp)): same 4-leaf average `C`, same anti-Hermitian `F = (C−C†)/2`, same three-term combination. Overall factor `−1/(2π²)` (twice the naive `−1/(4π²)`) to produce integer Q on smooth configurations.

---

### HMC Hamiltonian

```
H = S + K

S = β Σ_{s, μ<ν} ( 1 − Re Tr[U_{μν}(s)] / Nc )        [Wilson gauge action]
K = Σ_{s, μ}  Tr[ π_μ(s)² ]  =  (1/2) Σ_{s,μ,a} (π_μ^a)²
```

`π_μ(s)` are Hermitian traceless momenta (`su(3)` algebra, Hermitian convention). The equality `Tr[π²] = (1/2) Σ_a (π^a)²` follows from the generator normalisation `Tr[T_a T_b] = δ_{ab}/2`.

> **Note:** `hamiltonian_full` in `observables.cpp` is an independent copy of this formula for external use. The HMC accept/reject step internally calls `hamiltonian()` in `hmc.cpp`, which is the same expression.

---

### Clover Energy Density

**Lüscher's definition** ([arXiv:1006.4518](https://arxiv.org/abs/1006.4518), eq. 2.4):

```
E(t) = − (1/2V) Σ_{s, all μ,ν} tr[ G_{μν}(s,t)² ]
```

`G_{μν}` is **anti-Hermitian** (algebra-valued), with generator normalisation `tr[T_a T_b] = −δ_{ab}/2`,
so `tr[G²] ≤ 0` and `E ≥ 0`.

**Lattice discretisation** — clover field strength:

```
Q_{μν}(s) = (C_{μν}(s) − C†_{μν}(s)) / (8i)       [Hermitian, F in code]
C_{μν}    = sum of 4 oriented plaquettes (clover leaf)
```

**Factor-of-2 accounting:** The sum over all `μ,ν` (12 terms in 4D) equals `2 × Σ_{μ<ν}` by antisymmetry of `G`. With `G = i Q` (anti-Hermitian), `tr[G²] = −tr[Q²]`, giving:

```
E = −(1/2V) × 2 × Σ_{s, μ<ν} (−tr[Q²])  =  (1/V) Σ_{s, μ<ν} Tr[ Q_{μν}(s)² ]
```

This is exactly what the code computes. Used to define the hadronic scale `t₀` via:

```
t² E(t₀) = 0.3        [Lüscher-Weisz]
```

---

### Wilson Loop

```
W(r, t) = (1 / Nc V) Σ_s Re Tr[ R_{r×t}(s) ]
```

`R_{r×t}(s)` is the ordered product of links around an `r×t` rectangle in the `(μ, ν)` plane. Related to the static quark potential in the large-`t` limit:

```
V(r) = − lim_{t→∞} ln W(r, t) / t
```

---

## Smearing & Measurement Strategy

Each production trajectory follows this order:

```
1. hmc_step(gf)                      ← Markov chain advance on the UNSMEARED config
2. plaquette(gf)          → Plaq.csv ← raw plaquette on unsmeared config
3. wilson_loop(gf, r, t)  → WS_loop.csv ← Wilson loops on unsmeared config
4. run_flow(copy of gf)   → smear.csv   ← Wilson flow on a copy; original unchanged
```

### Wilson flow detail

```
run_flow(copy of gf, eps, t_max)   ← original gf NEVER modified
    for each flow step t:
        flow_step(copy, eps)
        Q(t), E(t)  →  smear.csv
```

**APE smearing** is kept as a library function (`src/ape.cpp`) but is not used in the production loop.

---

## Output

Results are written to `results/T{Lt}_S{Lx}/beta{beta}/flow/` by default (or the path given by `--out`):

| File | Columns | Written when |
|------|---------|--------------|
| `Plaq.csv` | `traj, plaq` | Once per trajectory (unsmeared config) |
| `WS_loop.csv` | `traj, r, t, W` | Once per trajectory × r (1..Lx) × t (1..Lt) |
| `smear.csv` | `traj, t_flow, Q, E` | Once per flow step × trajectory |
