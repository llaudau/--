# HMC_GPU - Lattice Gauge Theory Simulation on GPU

A CUDA implementation of **Hybrid Monte Carlo (HMC)** and **Metropolis** algorithms for simulating **SU(3) gauge fields** (Lattice Gauge Theory) on GPUs. This project is designed as a simplified version of QUDA (QCD GPU library) for educational and research purposes.

## Overview

This project implements numerical simulations of **pure gauge SU(3) theory** using Markov Chain Monte Carlo methods. It's commonly used in computational physics to study Quantum Chromodynamics (QCD) - the theory of strong interactions between quarks and gluons.

### Key Features

- **GPU-Accelerated**: All computations run on NVIDIA GPUs using CUDA
- **SU(3) Gauge Group**: 3×3 special unitary matrices (determinant = 1, U† = U⁻¹)
- **Two Monte Carlo Algorithms**:
  - Hybrid Monte Carlo (HMC) - molecular dynamics based
  - HMC with Dynamical Fermions - includes fermion determinant via pseudofermions
  - Metropolis-Hastings algorithm - local updates
- **Wilson Action**: Uses the Wilson plaquette action for gauge field evolution
- **Dirac Operator**: Wilson-Dirac operator for fermion interactions

## Project Structure

```
hmc_gpu/
├── CMakeLists.txt            # CMake build configuration
├── CMakePresets.json         # CMake presets for different compilers
├── include/structs/          # Header files for core data structures
│   ├── complex.cuh           # Custom complex number type (float/double)
│   ├── matrix.cuh            # Template-based N×N matrix class
│   ├── math_helper.cuh      # Mathematical helper functions
│   ├── gauge_operation.cuh  # Gauge field operations
│   ├── lattice.cuh          # GaugeField class and LatticeView
│   ├── spinor.cuh           # Spinor structures for fermions
│   └── dirac.cuh            # Dirac operator definitions
├── lib/kernels/              # CUDA kernel implementations
│   ├── lattice.cu           # Lattice initialization kernels
│   ├── update_hmc.cu        # HMC evolution kernels (gauge only)
│   ├── update_hmc_dynferm.cu # HMC with dynamical fermions
│   ├── update_Metro.cu      # Metropolis update kernels
│   ├── phy_obse.cu          # Physical observables (Hamiltonian, plaquette)
│   ├── dirac.cu             # Dirac operator kernels
│   ├── cg.cu                # CG solver for dynamical fermions
│   └── use.cu               # Utility kernels
├── tests/                    # Test executables
│   ├── test_metro.cu        # Metropolis algorithm test
│   ├── test_all.cu          # Comprehensive test suite
│   ├── test_complex_def.cpp # Complex number tests
│   ├── test_matrix_def.cu   # Matrix operations tests
│   ├── autocorrelate_study.cu # Autocorrelation study
│   └── autocorrelate_hmc.cu # HMC autocorrelation analysis
├── include/linalg/           # Linear algebra solvers
│   └── cg_solver.cuh        # CG solver for dynamical fermions
├── build/                    # CMake build directory
├── out/                     # Build output directory
└── results/                 # Simulation output data
    ├── hmc_results/         # HMC simulation results
    └── metro_results/       # Metropolis simulation results
```

## Physics Background

### What is Lattice Gauge Theory?

Lattice Gauge Theory discretizes spacetime into a 4D hypercubic lattice. Each link between neighboring lattice sites contains a gauge field variable (an SU(3) matrix). The theory is described by the **Wilson Action**:

$$S = \frac{\beta}{3} \sum_x \sum_{\mu<\nu} \text{Re}\,\text{Tr}[1 - U_{\mu\nu}(x)]$$

Where $U_{\mu\nu}(x)$ is the **plaquette** - the product of 4 gauge links forming a loop.

### Algorithms Implemented

#### 1. Hybrid Monte Carlo (HMC)

HMC combines molecular dynamics with Metropolis acceptance criterion:

1. **Refresh Momenta**: Draw new conjugate momenta from Gaussian distribution
2. **Molecular Dynamics**: Integrate Hamiltonian equations using leapfrog integration
   - Force calculation: $\dot{P} = -\frac{\partial S}{\partial U}$
   - Link update: $\dot{U} = P \cdot U$
3. **Accept/Reject**: Metropolis step based on Hamiltonian difference

#### 2. HMC with Dynamical Fermions

Extends HMC to include fermion effects via pseudofermions:

1. **Pseudofermion Fields**: $\phi = M^{1/2} \psi$ where $\psi$ are Gaussian random fields
2. **Molecular Dynamics**: Include fermion force from $S_{\text{fermion}} = \phi^\dagger M^{-1}\phi$
3. **Conjugate Gradient**: Solve $M x = \phi$ for fermion force calculation
4. **Accept/Reject**: Full Hamiltonian including fermion contribution

#### 3. Metropolis-Hastings Algorithm

Local update algorithm for individual gauge links:
1. Propose new link: $U' = U \cdot V$ where V is random SU(3) near identity
2. Calculate action difference: $\Delta S = S[U'] - S[U]$
3. Accept with probability: $P = \min(1, e^{-\Delta S})$

## Core Data Structures

### Complex Numbers (`complex.cuh`)
- Template-based complex number type supporting `float` and `double`
- Optimized for GPU execution with `__host__ __device__` functions
- Inherits from CUDA's `float2`/`double2` for memory efficiency

### Matrices (`matrix.cuh`)
- Template-based N×N matrix class
- Special support for 3×3 matrices (SU(3))
- Key operations:
  - Matrix multiplication
  - Conjugate transpose (dagger)
  - Trace
  - Determinant

### Gauge Field (`lattice.cuh`)
- Manages GPU memory for gauge links and momenta
- Implements neighbor indexing for periodic boundary conditions
- Contains `GaugeField` class with methods for:
  - Initialization
  - HMC evolution (pure gauge and with fermions)
  - Metropolis updates
  - Observable calculation

### Spinors (`spinor.cuh`)
- Spinor structures for fermion fields
- 4-component spinors in 4D spacetime
- Dirac gamma matrix operations

### Dirac Operator (`dirac.cuh`)
- Wilson-Dirac operator implementation
- Support for even-odd preconditioning
- Fermion force calculation for HMC with dynamical fermions

## Building

```bash
# Using CMake with default settings
cmake --preset default
cmake --build build

# Run tests
./out/build/test_all
./out/build/test_metro
./out/build/autocorrelate_study
./out/build/autocorrelate_hmc
```

## Usage Example

```cpp
#include "lattice.cuh"

int main() {
    // Initialize lattice: 4×4×4×6 spatial-temporal lattice at β=6.0
    GaugeField lattice(4, 4, 4, 6, 6.0);

    // Calculate initial plaquette
    num_type plaq = lattice.calculate_plaquette();
    std::cout << "Initial plaquette: " << plaq << std::endl;

    // Run HMC simulation (pure gauge)
    auto plaquette_history = lattice.full_update(
        thermal_steps=50,    // Thermalization
        ntraj=100,            // Production trajectories
        interval=1,          // Measurements per trajectory
        length=1.0,           // MD trajectory length
        num_steps=10          // MD steps per trajectory
    );

    return 0;
}
```

### Test Programs

- `./out/build/autocorrelate_study` - Analyze autocorrelation times
- `./out/build/autocorrelate_hmc` - HMC autocorrelation analysis
- `./out/build/test_metro` - Metropolis algorithm test
- `./out/build/test_all` - Comprehensive test suite

## Observables

### Plaquette
The average plaquette measures the "smoothness" of the gauge field:
$$\langle P \rangle = \frac{1}{3} \text{Re}\,\text{Tr}[U_{\mu\nu}]$$

### Hamiltonian

Pure gauge:
$$H = S_{\text{Wilson}} + \sum_{\mu,x} \text{Tr}[P_\mu(x)^2]$$

With dynamical fermions:
$$H = S_{\text{Wilson}} + S_{\text{fermion}} + \sum_{\mu,x} \text{Tr}[P_\mu(x)^2]$$

Where $P_\mu(x)$ are conjugate momenta and $S_{\text{fermion}} = \phi^\dagger M^{-1}\phi$ is the pseudofermion action.

## Fermion Force Derivation

### Pseudofermion Action

The fermion determinant $\det(D)$ is incorporated via the pseudofermion method:

$$S_f[\phi] = \phi^\dagger (D D^\dagger)^{-1} \phi$$

where the pseudofermion field is generated as:
$$\phi = D^\dagger \eta, \quad \eta \sim \mathcal{N}(0, 1)$$

### Solving the Linear System

Define the solution field $\chi$:
$$\chi = (D D^\dagger)^{-1} \phi$$

which satisfies:
$$D D^\dagger \, \chi = \phi$$

The action becomes:
$$S_f = \phi^\dagger \chi = \chi^\dagger \phi$$

### Force Derivation

The force on link $U_\mu(x)$ in the SU(3) algebra basis is:

$$F_{\text{fermion}} = \sum_{i=1}^{8} T_i \, \nabla^i\left(\phi^\dagger (D D^\dagger)^{-1} \phi\right)$$

where:
- $T_i$ ($i=1,\dots,8$) are the SU(3) generators (Gell-Mann matrices divided by 2)
- $\nabla^i$ denotes the derivative with respect to the SU(3) link parameter $\omega_\mu^{(i)}$
- The generators satisfy: $\text{Tr}[T_i T_j] = \frac{1}{2} \delta_{ij}$

Using the chain rule and the linear system $D D^\dagger \chi = \phi$:

$$\boxed{F_{\text{fermion}} = -\sum_{i=1}^{8} T_i \, \chi^\dagger \left( \frac{\partial D}{\partial \omega^{(i)}} D^\dagger + D \frac{\partial D^\dagger}{\partial \omega^{(i)}} \right) \chi}$$

### Wilson-Dirac Operator Derivative

The Wilson-Dirac operator:
$$(D\psi)_x = \psi_x - \kappa \sum_\nu \left[(1+\gamma_\nu) U_\nu(x) \psi_{x+\nu} + (1-\gamma_\nu) U_\nu^\dagger(x-\nu) \psi_{x-\nu}\right]$$

Express the SU(3) link as $U_\mu(x) = \exp\left(i \sum_{a=1}^{8} \omega_\mu^{(a)}(x) T_a\right)$.

The derivative with respect to link parameters:
$$\frac{\partial D(n|m)}{\partial \omega_\mu^{(i)}} = -i\kappa(1-\gamma_\mu) T_i U_\mu(k)\,\delta_{n+\mu,m}\,\delta_{n,k} + i\kappa(1+\gamma_\mu) T_i U_\mu(k)^\dagger\,\delta_{n-\mu,m}\,\delta_{m,k}$$

### Final Fermion Force

After substituting the Dirac operator derivative, the force simplifies to contributions from forward and backward links:

$$F_\mu(x) = \text{traceless part of: } \left[ \chi^\dagger(x) (1+\gamma_\mu) \chi(x+\hat{\mu}) + \chi^\dagger(x-\hat{\mu}) (1-\gamma_\mu) \chi(x) \right]$$

This force is then projected onto the SU(3) algebra basis to obtain the 8 components $F_\mu^{(i)}(x)$.

### CG Solver Requirement

To compute the force, first solve:
$$(D D^\dagger) \, \chi = \phi$$

using the CG solver (`solve_cg_mdaggerm` in `lib/kernels/cg.cu`).

### Implementation Summary

1. **Generate pseudofermion**: $\phi = D^\dagger \eta$ with $\eta \sim \mathcal{N}(0,1)$
2. **Solve CG**: Compute $\chi = (D D^\dagger)^{-1} \phi$
3. **Compute force**: Evaluate $\chi^\dagger (1\pm\gamma_\mu) \chi$ terms
4. **Project to SU(3) basis**: Extract 8 components of the force
5. **Update momentum**: $P_\mu(x) \leftarrow P_\mu(x) - \epsilon \, F_\mu(x)$

## Key Implementation Details

1. **Memory Layout**: Links stored as `volume × 4` matrix array (AOS layout)
2. **Neighbor Table**: Precomputed neighbor indices for efficient GPU access
3. **Reunitarization**: Ensures SU(3) constraint after link updates
4. **Force Calculation**: Fermion force via CG solver and SU(3) generator projection
5. **Random Number Generation**: cuRAND for parallel random number generation
6. **Even-Odd Preconditioning**: Reduced fermion matrix condition number (future)
7. **Pseudofermions**: Stochastic representation of fermion determinant
8. **CG Solver**: Conjugate Gradient for solving $(D D^\dagger)\chi = \phi$

## Simulation Parameters

Typical parameters used in simulations:
- **Lattice sizes**: 4⁴, 8⁴, 10⁴
- **β values**: 5.7-6.3 (strong coupling to weak coupling transition)
- **MD step size**: ε = 0.1-0.2
- **Trajectory length**: τ = 0.5-1.0

## References

- **MILC Code**: Original lattice QCD code reference
- **QUDA**: Production-quality QCD library
- **"Lattice Gauge Theory" by Creutz**: Standard textbook
- **"Introduction to Lattice QCD" by DeGrand & DeTar**: Practical introduction
- **"Quantum Fields on the Computer" by Montvay & Münster**: Advanced topics

## Performance Notes

- Optimized for NVIDIA GPUs with compute capability ≥ 7.0
- Uses separate compilation for faster builds
- `#pragma unroll` directives for loop optimization
- Register caching for frequently accessed data
- Shared memory usage for stencil operations
- Fermion conjugate gradient solver optimization

## License

This project is for educational and research purposes.
