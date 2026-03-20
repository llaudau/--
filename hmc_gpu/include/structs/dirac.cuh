#pragma once
#include "spinor.cuh"
#include "matrix.cuh"
namespace qcdcuda {

// Wilson Dirac operator - clean implementation
// Only keeps the essential Wilson hopping function

// Wilson hopping term: applies SU(3) gauge link and gamma matrix to a spinor
// Computes: chi += κ × (1 + sign×γ_μ) × U × ψ
//
// Parameters:
//   chi  - Output spinor (accumulated result)
//   U    - SU(3) gauge link matrix (3×3)
//   psi  - Input spinor at neighbor site
//   mu   - Direction: 0=x, 1=y, 2=z, 3=t
//   kappa - Wilson hopping parameter
//   sign - +1 for forward direction (1 + γ_μ)
//         -1 for backward direction (1 - γ_μ)
template<typename T >
__device__ __host__ inline void wilson_hop(Spinor<complex<T>>& chi,
                                           const Matrix<complex<T>, 3>& U,
                                           const Spinor<complex<T>>& psi,
                                           int mu,
                                           T kappa,
                                           int sign) {
    // chi += κ × (1 + sign×γ_μ) × U × ψ
    
    #pragma unroll
    for(int c = 0; c < 3; c++) {
        #pragma unroll
        for(int s_out = 0; s_out < 4; s_out++) {
            // Identity term: κ × U × ψ
            #pragma unroll
            for(int d = 0; d < 3; d++) {
                chi(c, s_out) += kappa * U(c, d) * psi(d, s_out);
            }
            
            // Gamma term with sign: sign×κ × γ_μ × U × ψ
            #pragma unroll
            for(int s_in = 0; s_in < 4; s_in++) {
                int gamma_val = gamma_matrices::gamma_elem(mu, s_out, s_in);
                if (gamma_val != 0) {
                    #pragma unroll
                    for(int d = 0; d < 3; d++) {
                        complex<T> gamma_factor;
                        int signed_gamma = sign * gamma_val;
                        if (signed_gamma == 1) {
                            gamma_factor = complex<T>(1, 0);
                        } else if (signed_gamma == -1) {
                            gamma_factor = complex<T>(-1, 0);
                        } else if (signed_gamma == 3) {
                            gamma_factor = complex<T>(0, 1);
                        } else {  // -3
                            gamma_factor = complex<T>(0, -1);
                        }
                        chi(c, s_out) += kappa * gamma_factor * U(c, d) * psi(d, s_in);
                    }
                }
            }
        }
    }
}

// Apply gamma_5 (chirality projection)
// ψ_out = γ_5 × ψ_in
template<typename T>
__device__ __host__ inline void apply_gamma5(Spinor<complex<T>>& psi_out, 
                                            const Spinor<complex<T>>& psi_in) {
    #pragma unroll
    for(int c = 0; c < 3; c++) {
        #pragma unroll
        for(int s_out = 0; s_out < 4; s_out++) {
            complex<T> sum = complex<T>(0.0, 0.0);
            #pragma unroll
            for(int s_in = 0; s_in < 4; s_in++) {
                int g5 = gamma_matrices::gamma5(s_out, s_in);
                if (g5 != 0) {
                    sum += complex<T>(g5, 0) * psi_in(c, s_in);
                }
            }
            psi_out(c, s_out) = sum;
        }
    }
}

} // namespace qcdcuda
