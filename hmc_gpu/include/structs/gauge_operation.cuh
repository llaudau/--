#pragma once
#include "matrix.cuh"
#include "complex.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

//in my definition e^{i epsi P} is a su3, so P is hermitian traceless, and the make_hermi function can project force to this space



namespace qcdcuda{
    template<typename T>
    __device__ inline void reunitarize(Matrix<complex<T>, 3> &m) {
    T row0_norm = gpu_rsqrt(norm(m(0,0)) + norm(m(0,1)) + norm(m(0,2)));
    T inv_norm0 = T(1.0) * row0_norm;
    m(0,0) = m(0,0) * inv_norm0;
    m(0,1) = m(0,1) * inv_norm0;
    m(0,2) = m(0,2) * inv_norm0;
    complex<T> dot01 = conj(m(0,0)) * m(1,0) + conj(m(0,1)) * m(1,1) + conj(m(0,2)) * m(1,2);
    m(1,0) = m(1,0) - dot01 * m(0,0);
    m(1,1) = m(1,1) - dot01 * m(0,1);
    m(1,2) = m(1,2) - dot01 * m(0,2);

    T row1_norm = gpu_rsqrt(norm(m(1,0)) + norm(m(1,1)) + norm(m(1,2)));
    T inv_norm1 = T(1.0) * row1_norm;
    m(1,0) = m(1,0) * inv_norm1;
    m(1,1) = m(1,1) * inv_norm1;
    m(1,2) = m(1,2) * inv_norm1;
    m(2,0) = conj(m(0,1) * m(1,2) - m(0,2) * m(1,1));
    m(2,1) = conj(m(0,2) * m(1,0) - m(0,0) * m(1,2));
    m(2,2) = conj(m(0,0) * m(1,1) - m(0,1) * m(1,0));

}
    template<typename T,int N>
    __device__ inline void make_antihermi(Matrix<T,N> &m){
        Matrix<T,N> am = m +complex<float>(-1.0,0.0)* conj(m);

        T imag_trace =T(0.0,0.0);
        #pragma unroll
        for (int i=0; i<N; i++) imag_trace = imag_trace+am(i,i);
        #pragma unroll
        for (int i=0; i<N; i++) {
            am(i,i) =am(i,i)+ imag_trace*T(-1/N,0.0);
        }
        m = T(0.5,0.0)* am;
    }
    template<typename T,int N>
    __device__ inline void make_hermi(Matrix<T,N> &m){

        Matrix<T,N> am = m + conj(m);

        T imag_trace =T(0.0,0.0);
        #pragma unroll
        for (int i=0; i<N; i++) imag_trace = imag_trace+am(i,i);
        #pragma unroll
        for (int i=0; i<N; i++) {
            am(i,i) =am(i,i)+ imag_trace*T(-1/N,0.0);
        }
        m = T(0.5,0.0)* am;
    }

    template <typename T>
__device__ void generate_random_su3_near_identity(Matrix<T, 3> &U, float epsilon, curandState *state) {
    U.setIdentity();

    int planes[3][2] = {{0, 1}, {1, 2}, {0, 2}};

    for (int p = 0; p < 3; p++) {
        int i = planes[p][0];
        int j = planes[p][1];


        float a1 = (curand_uniform(state) - 0.5f) * epsilon;
        float a2 = (curand_uniform(state) - 0.5f) * epsilon;
        float a3 = (curand_uniform(state) - 0.5f) * epsilon;
        float a0 = sqrt(1.0f - (a1*a1 + a2*a2 + a3*a3));

        T r00 = T(a0,  a3); // a0 + i*a3
        T r01 = T(a2,  a1); // a2 + i*a1
        T r10 = T(-a2, a1); // -a2 + i*a1
        T r11 = T(a0, -a3); // a0 - i*a3

        for (int k = 0; k < 3; k++) {
            T tmp_ik = r00 * U(i, k) + r01 * U(j, k);
            T tmp_jk = r10 * U(i, k) + r11 * U(j, k);
            U(i, k) = tmp_ik;
            U(j, k) = tmp_jk;
        }
    }
    
    // Always reunitarize to fix floating point drift
    reunitarize(U);
    }




    template <typename T>
    // i'm not sure whether this is correct but chi square is correct, so i will use it now 
__device__ void generate_gaussian_su3_algebra(Matrix<T, 3> &P, curandState *state) {
    // const float inv_sqrt2 = 0.707106781f; // 1/sqrt(2) to adjust variance
    const float inv_sqrt2 = 0.5; // 1/sqrt(2) to adjust variance
    
    const float inv_sqrt3 = 0.577350269;


    float upper1=curand_normal(state);
    float upper2=curand_normal(state);
    float upper3=curand_normal(state);
    float upper4=curand_normal(state);
    float upper5=curand_normal(state);
    float upper6=curand_normal(state);
    
    T p01 = T(upper1, -upper2) * inv_sqrt2 ;
    T p02 = T(upper3, -upper4) * inv_sqrt2 ;
    T p12 = T(upper5, -upper6) * inv_sqrt2 ;

    P(0, 1) = p01; P(1, 0) = conj(p01);
    P(0, 2) = p02; P(2, 0) = conj(p02);
    P(1, 2) = p12; P(2, 1) = conj(p12);


    float d0 = curand_normal(state)* inv_sqrt2;
    float d1 = curand_normal(state)* inv_sqrt2*inv_sqrt3;
    
    P(0, 0) = T((d0 +d1) , 0.0f);
    P(1, 1) = T((-d0 + d1) , 0.0f);
    P(2, 2) = T(( - 2*d1) , 0.0f);
}
}