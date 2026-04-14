# HMC MPI/CPU Implementation Plan for Tianhe (ARM64)

## Overview
Port `hmc_gpu` (CUDA) to a pure CPU/MPI+OpenMP version for Tianhe ARM64 nodes.
Double precision only. No fermion code. Includes HMC, observables, Wilson gradient flow.

## Directory Structure
```
hmc/
├── CMakeLists.txt
├── run.slurm
├── plan/plan.md
├── include/
│   ├── complex.hpp        # complex<double>, no CUDA
│   ├── matrix.hpp         # Matrix<double,3>, no CUDA
│   ├── gauge_ops.hpp      # reunitarize, Gaussian SU3 momenta
│   ├── params.hpp         # SimParams + CLI parser
│   ├── lattice.hpp        # GaugeField class (MPI-distributed)
│   ├── hmc.hpp
│   ├── observables.hpp
│   └── gradient_flow.hpp
├── src/
│   ├── lattice.cpp
│   ├── hmc.cpp
│   ├── observables.cpp
│   ├── gradient_flow.cpp
│   └── io.cpp
└── apps/
    └── run.cpp
```

## Design Decisions
- **MPI domain decomp**: split T-direction; each rank owns Lt/nranks time slices + 1-layer ghost
- **OpenMP**: parallelize site loops within rank; 8 OMP threads/rank, 8 MPI/node (NUMA aligned)
- **RNG**: std::mt19937_64, seed = base_seed + mpi_rank
- **Gradient flow**: Wilson flow (replaces APE smearing), Euler integration, step ε=0.01
- **Topological charge**: clover definition, computed after gradient flow
- **Memory layout**: AOS (Array of Structures), consistent with GPU version
- **No float path**: using num_t = double everywhere

## Parameters (SimParams)
- Lx=Ly=Lz=10, Lt=16, beta=6.3
- traj_len=0.2, md_steps=40
- thermal=500, n_samples=10000
- flow_eps=0.01, flow_t_max=1.0
- nodes=1 (adjustable up to 32)

## Implementation Steps
1. Create dirs + plan ← current
2. complex.hpp, matrix.hpp
3. gauge_ops.hpp, params.hpp
4. lattice.hpp, lattice.cpp (MPI init, halo exchange, hopping table)
5. hmc.hpp, hmc.cpp (leapfrog, force, accept/reject)
6. observables.hpp, observables.cpp (plaquette, H, Q with MPI_Allreduce)
7. gradient_flow.hpp, gradient_flow.cpp (Wilson flow)
8. io.cpp
9. CMakeLists.txt
10. apps/run.cpp
11. Compile + test on small lattice
