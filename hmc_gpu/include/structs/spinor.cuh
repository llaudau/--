#pragma once
#include "complex.cuh"
#include <cuda_runtime.h>

namespace qcdcuda {

// =============================================================================
// Spinor class: 3 colors × 4 spin components = 12 complex numbers
// Layout: data[color * 4 + spin]
// =============================================================================

template<typename T>
class Spinor {
public:
    T data[12];

    __host__ __device__ inline Spinor() { setZero(); }

    __host__ __device__ inline void setZero() {
        #pragma unroll
        for(int i=0; i<12; i++) data[i] = T(0.0);
    }

    __host__ __device__ inline void setIdentity() {
        setZero();
        #pragma unroll
        for(int c=0; c<3; c++) {
            data[c*4 + 0] = T(1.0, 0.0);
        }
    }

    __host__ __device__ inline T& operator()(int color, int spin) {
        return data[color * 4 + spin];
    }

    __host__ __device__ inline const T& operator()(int color, int spin) const {
        return data[color * 4 + spin];
    }

    __host__ __device__ inline T& operator[](int i) { return data[i]; }

    __host__ __device__ inline const T& operator[](int i) const { return data[i]; }

    __host__ __device__ constexpr static int size() { return 12; }

    __host__ __device__ inline void add(const Spinor<T>& other, T scale) {
        #pragma unroll
        for(int i=0; i<12; i++) data[i] += other.data[i] * scale;
    }

    __host__ __device__ inline void copy(const Spinor<T>& other) {
        #pragma unroll
        for(int i=0; i<12; i++) data[i] = other.data[i];
    }
};

// =============================================================================
// Basic spinor operations
// =============================================================================

template<typename T>
__device__ __host__ inline Spinor<T> operator+(const Spinor<T>& a, const Spinor<T>& b) {
    Spinor<T> res;
    #pragma unroll
    for(int i=0; i<12; i++) res[i] = a[i] + b[i];
    return res;
}

template<typename T>
__device__ __host__ inline Spinor<T> operator-(const Spinor<T>& a, const Spinor<T>& b) {
    Spinor<T> res;
    #pragma unroll
    for(int i=0; i<12; i++) res[i] = a[i] - b[i];
    return res;
}

template<typename T>
__device__ __host__ inline Spinor<T> operator*(const Spinor<T>& a, const T& scale) {
    Spinor<T> res;
    #pragma unroll
    for(int i=0; i<12; i++) res[i] = a[i] * scale;
    return res;
}

template<typename T>
__device__ __host__ inline Spinor<T> operator*(const T& scale, const Spinor<T>& a) {
    return a * scale;
}

template<typename T>
__device__ __host__ inline void operator+=(Spinor<T>& a, const Spinor<T>& b) {
    #pragma unroll
    for(int i=0; i<12; i++) a[i] += b[i];
}

template<typename T>
__device__ __host__ inline void operator*=(Spinor<T>& a, const T& scale) {
    #pragma unroll
    for(int i=0; i<12; i++) a[i] *= scale;
}

// =============================================================================
// Inner products and norms
// =============================================================================

// Spinor inner product: Σ_c,s ψ₁^*(c,s) × ψ₂(c,s)
// Result is complex scalar: ψ₁^† × ψ₂
template<typename T>
__device__ __host__ inline complex<T> spinor_inner_product(const Spinor<T>& psi1, const Spinor<T>& psi2) {
    complex<T> result = complex<T>(0.0, 0.0);
    #pragma unroll
    for(int i=0; i<12; i++) {
        result += conj(psi1[i]) * psi2[i];
    }
    return result;
}

// Spinor norm squared: Σ_c,s |ψ(c,s)|² = ψ^† × ψ
template<typename T>
__device__ __host__ inline T spinor_norm_sq(const Spinor<T>& psi) {
    T result = T(0.0);
    #pragma unroll
    for(int i=0; i<12; i++) {
        result += norm(psi[i]);
    }
    return result;
}

// =============================================================================
// Gamma matrices in Euclidean space (chiral representation)
// γ_μ² = 1, {γ_μ, γ_ν} = 0 for μ ≠ ν
// =============================================================================

enum GammaMatrixIndex {
    GAMMA_X = 0,
    GAMMA_Y = 1,
    GAMMA_Z = 2,
    GAMMA_T = 3
};

namespace gamma_matrices {

// γ_1 (x-direction)
__device__ __host__ inline constexpr int gamma1(int s_out, int s_in) {
    // γ₁ = [[ 0,  0,  0, -i],
    //       [ 0,  0, -i,  0],
    //       [ 0,  i,  0,  0],
    //       [ i,  0,  0,  0]]
    if (s_out == 0 && s_in == 3) return 3;  // -i
    if (s_out == 1 && s_in == 2) return 3;  // -i
    if (s_out == 2 && s_in == 1) return 1;  // +i
    if (s_out == 3 && s_in == 0) return 1;  // +i
    return 0;
}

// γ_2 (y-direction)  
__device__ __host__ inline constexpr int gamma2(int s_out, int s_in) {
    // γ₂ = [[ 0,  0,  0, -1],
    //       [ 0,  0,  1,  0],
    //       [ 0,  1,  0,  0],
    //       [-1,  0,  0,  0]]
    if (s_out == 0 && s_in == 3) return -1;  // -1
    if (s_out == 1 && s_in == 2) return 1;   // +1
    if (s_out == 2 && s_in == 1) return 1;   // +1
    if (s_out == 3 && s_in == 0) return -1;  // -1
    return 0;
}

// γ_3 (z-direction)
__device__ __host__ inline constexpr int gamma3(int s_out, int s_in) {
    // γ₃ = [[ 0,  0, -i,  0],
    //       [ 0,  0,  0,  i],
    //       [ i,  0,  0,  0],
    //       [ 0, -i,  0,  0]]
    if (s_out == 0 && s_in == 2) return 3;  // -i
    if (s_out == 1 && s_in == 3) return 1;  // +i
    if (s_out == 2 && s_in == 0) return 1;  // +i
    if (s_out == 3 && s_in == 1) return 3;  // -i
    return 0;
}

// γ_4 (t-direction, γ_0 in Minkowski)
__device__ __host__ inline constexpr int gamma4(int s_out, int s_in) {
    // γ₄ = [[ 0,  0,  1,  0],
    //       [ 0,  0,  0,  1],
    //       [ 1,  0,  0,  0],
    //       [ 0,  1,  0,  0]]
    if (s_out == 0 && s_in == 2) return 1;   // +1
    if (s_out == 1 && s_in == 3) return 1;  // +1
    if (s_out == 2 && s_in == 0) return 1;  // +1
    if (s_out == 3 && s_in == 1) return 1;   // +1
    return 0;
}

// Get gamma matrix element for direction mu
// Returns: 0 for zero, +1 for +1, -1 for -1, +3 for +i, -3 for -i
__device__ __host__ inline constexpr int gamma_elem(int mu, int s_out, int s_in) {
    switch(mu) {
        case 0: return gamma1(s_out, s_in);
        case 1: return gamma2(s_out, s_in);
        case 2: return gamma3(s_out, s_in);
        case 3: return gamma4(s_out, s_in);
        default: return 0;
    }
}

// γ_5 = γ₁γ₂γ₃γ₄ (chirality matrix)
// γ_5 = [[ 1,  0, 0,  0],
//        [ 0,  1,  0, 0],
//        [0,  0,  -1,  0],
//        [ 0, 0,  0,  -1]]
__device__ __host__ inline constexpr int gamma5(int s_out, int s_in) {
    if (s_out == 0 && s_in == 0) return 1;  // 1
    if (s_out == 1 && s_in == 1) return 1;  // 1
    if (s_out == 2 && s_in == 2) return -1;  // -1
    if (s_out == 3 && s_in == 3) return -1;  // -1
    return 0;
}

} // namespace gamma_matrices

// =============================================================================
// Utility operations
// =============================================================================

// Apply γ_5 (chirality projection)
// ψ_out = γ_5 × ψ_in
template<typename T>
__device__ __host__ inline void apply_gamma5(Spinor<T>& psi_out, const Spinor<T>& psi_in) {
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

// Pointwise spinor multiplication (Hadamard product)
// χ_c,s = ψ₁_c,s ⊙ ψ₂_c,s
template<typename T>
__device__ __host__ inline void spinor_pointwise_mul(Spinor<T>& result,
                                                      const Spinor<T>& a,
                                                      const Spinor<T>& b) {
    #pragma unroll
    for(int i = 0; i < 12; i++) {
        result[i] = a[i] * b[i];
    }
}

// Scale spinor by real scalar
template<typename T>
__device__ __host__ inline void spinor_scale(Spinor<T>& psi, T scale) {
    #pragma unroll
    for(int i = 0; i < 12; i++) {
        psi[i] *= scale;
    }
}

} // namespace qcdcuda
