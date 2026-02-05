# HMC_GPU - Lattice Gauge Theory Simulation on GPU

A CUDA implementation of **Hybrid Monte Carlo (HMC)** and **Metropolis** algorithms for simulating **SU(3) gauge fields** (Lattice Gauge Theory) on GPUs. This project is designed as a simplified version of QUDA (QCD GPU library) for educational and research purposes.

## Overview

This project implements numerical simulations of **pure gauge SU(3) theory** using Markov Chain Monte Carlo methods. It's commonly used in computational physics to study Quantum Chromodynamics (QCD) - the theory of strong interactions between quarks and gluons.

### Key Features

- **GPU-Accelerated**: All computations run on NVIDIA GPUs using CUDA
- **SU(3) Gauge Group**: 3×3 special unitary matrices (determinant = 1, U† = U⁻¹)
- **Two Monte Carlo Algorithms**:
  - Hybrid Monte Carlo (HMC) - molecular dynamics based
  - Metropolis-Hastings algorithm - local updates
- **Wilson Action**: Uses the Wilson plaquette action for gauge field evolution

## Project Structure

```
hmc_gpu/
├── CMakeLists.txt           # CMake build configuration
├── CMakePresets.json        # CMake presets for different compilers
├── notes.txt                # Development notes and design decisions
├── include/structs/         # Header files for core data structures
│   ├── lattice.cuh         # GaugeField class and LatticeView
│   ├── complex.cuh         # Custom complex number type (float/double)
│   ├── matrix.cuh           # Template-based N×N matrix class
│   ├── math_helper.cuh     # Mathematical helper functions
│   └── gauge_operation.cuh # Gauge field operations
├── lib/kernels/             # CUDA kernel implementations
│   ├── lattice.cu          # Lattice initialization kernels
│   ├── update_hmc.cu       # HMC evolution kernels
│   ├── update_Metro.cu     # Metropolis update kernels
│   ├── phy_obse.cu         # Physical observables (Hamiltonian, plaquette)
│   └── use.cu              # Utility kernels
├── tests/                   # Test executables
│   ├── test_metro.cu       # Main test with Metropolis algorithm
│   ├── test_all.cu         # Comprehensive test suite
│   ├── test_complex_def.cpp # Complex number tests
│   └── test_matrix_def.cu  # Matrix operations tests
├── build/                   # CMake build directory
├── out/                     # Build output directory
└── results/                # Simulation output data
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

#### 2. Metropolis-Hastings Algorithm

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
  - HMC evolution
  - Metropolis updates
  - Observable calculation

## Building

```bash
# Using CMake with default settings
cmake --preset default
cmake --build build

# Run tests
./out/build/GCC\ 13.3.0/x86_64-linux-gnu/complex_test
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

    // Run HMC simulation
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

## Observables

### Plaquette
The average plaquette measures the "smoothness" of the gauge field:
$$\langle P \rangle = \frac{1}{3} \text{Re}\,\text{Tr}[U_{\mu\nu}]$$

### Hamiltonian
$$H = S_{\text{Wilson}} + \sum_{\mu,x} \text{Tr}[P_\mu(x)^2]$$

Where $P_\mu(x)$ are conjugate momenta.

## Key Implementation Details

1. **Memory Layout**: Links stored as `volume × 4` matrix array (AOS layout)
2. **Neighbor Table**: Precomputed neighbor indices for efficient GPU access
3. **Reunitarization**: Ensures SU(3) constraint after link updates
4. **Force Calculation**: Uses traceless anti-hermitian projection
5. **Random Number Generation**: cuRAND for parallel random number generation

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

## Performance Notes

- Optimized for NVIDIA GPUs with compute capability ≥ 7.0
- Uses separate compilation for faster builds
- `#pragma unroll` directives for loop optimization
- Register caching for frequently accessed data

## License

This project is for educational and research purposes.
