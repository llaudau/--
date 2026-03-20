#include "lattice.cuh"
#include "dirac.cuh"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

// Wilson Dslash kernel - applies the full Dirac operator to all lattice sites
// Computes: Dslash × ψ for the entire lattice
namespace qcdcuda{
__global__ void kernel_wilson_dslash(LatticeView lat, FermionFieldView src, FermionFieldView dst, int is_dagger) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lat.volume) return;

    Spinor<complex<num_type>> chi;
    chi.setZero();

    for (int mu = 0; mu < 4; mu++) {
        // Forward direction: x -> x+μ
        int site_fwd = lat.neighbor(site, mu);
        Spinor<complex<num_type>>& psi_fwd = src(site_fwd);
        Matrix<complex<num_type>, 3> U_fwd;
        
        if (!is_dagger) {
            U_fwd = lat.d_links[lat.link_idx(site, mu)];
        } else {
            U_fwd = lat.d_links[lat.link_idx(site_fwd, mu)].dagger();
        }
        
        // Forward hopping: (1 + γ_μ)
        wilson_hop(chi, U_fwd, psi_fwd, mu, lat.kappa, +1);
        
        // Backward direction: x -> x-μ
        int site_bwd = lat.neighbor(site, mu + 4);
        Spinor<complex<num_type>>& psi_bwd = src(site_bwd);
        Matrix<complex<num_type>, 3> U_bwd;
        
        if (!is_dagger) {
            U_bwd = lat.d_links[lat.link_idx(site_bwd, mu)].dagger();
        } else {
            U_bwd = lat.d_links[lat.link_idx(site, mu)];
        }
        
        // Backward hopping: (1 - γ_μ)
        wilson_hop(chi, U_bwd, psi_bwd, mu, lat.kappa, -1);
    }

    dst(site).copy(chi);
}

// Optimized Wilson Dslash - better memory coalescing
// Pre-loads all 4 directions before applying hopping
__global__ void kernel_wilson_dslash_optimized(LatticeView lat, FermionFieldView src, FermionFieldView dst, int is_dagger) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lat.volume) return;

    Spinor<complex<num_type>> chi;
    chi.setZero();

    // Load all forward links and spinors at once
    Matrix<complex<num_type>, 3> U_fwd[4];
    Spinor<complex<num_type>> psi_fwd[4];
    
    #pragma unroll
    for (int mu = 0; mu < 4; mu++) {
        int s = lat.neighbor(site, mu);
        psi_fwd[mu] = src(s);
        
        if (!is_dagger) {
            U_fwd[mu] = lat.d_links[lat.link_idx(site, mu)];
        } else {
            U_fwd[mu] = lat.d_links[lat.link_idx(s, mu)].dagger();
        }
    }

    // Apply all 4 forward directions: (1 + γ_μ)
    #pragma unroll
    for (int mu = 0; mu < 4; mu++) {
        wilson_hop(chi, U_fwd[mu], psi_fwd[mu], mu, lat.kappa, +1);
    }

    // Load all backward links and spinors
    Matrix<complex<num_type>, 3> U_bwd[4];
    Spinor<complex<num_type>> psi_bwd[4];
    
    #pragma unroll
    for (int mu = 0; mu < 4; mu++) {
        int s = lat.neighbor(site, mu + 4);
        psi_bwd[mu] = src(s);
        
        if (!is_dagger) {
            U_bwd[mu] = lat.d_links[lat.link_idx(s, mu)].dagger();
        } else {
            U_bwd[mu] = lat.d_links[lat.link_idx(site, mu)];
        }
    }

    // Apply all 4 backward directions: (1 - γ_μ)
    #pragma unroll
    for (int mu = 0; mu < 4; mu++) {
        wilson_hop(chi, U_bwd[mu], psi_bwd[mu], mu, lat.kappa, -1);
    }

    dst(site).copy(chi);
}

// Fermion action calculation: S = Σ_x φ^dagger(x) × φ(x)
// This is a simplified pseudofermion action (full version requires CG solver)
__global__ void kernel_fermion_action(FermionFieldView phi, num_type* result) {
    extern __shared__ num_type sdata[];

    int tid = threadIdx.x;
    int site = blockIdx.x * blockDim.x + threadIdx.x;

    num_type local_norm = 0.0;
    if (site < phi.volume) {
        Spinor<complex<num_type>>& psi = phi(site);
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            local_norm += norm(psi[i]);
        }
    }

    sdata[tid] = local_norm;
    __syncthreads();

    // Parallel reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

// Placeholder for fermion force calculation
// Will be implemented when needed for full HMC with fermions
__global__ void kernel_fermion_force(LatticeView lat, FermionFieldView phi, num_type factor) {
    // TODO: Implement fermion force for HMC
    // The force is: F_μ(x) = ∂S_f/∂U_μ(x)
    // Requires computing χ = (D^dagger D)^{-1} φ then differentiating
}
}
