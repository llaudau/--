#pragma once
#include "complex.cuh"
#include "math_helper.cuh"
#include <cuda_runtime.h>

namespace qcdcuda{
    template<typename T, int N>
class Matrix {
public:
    // 1. Data Storage: Flat array is best for GPU registers
    T data[N * N];

    // 2. Essential Constructors
    Matrix() = default; 

    
    __host__ __device__ static Matrix identity() {
        Matrix identity_matrix;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                // Assuming your complex type can be constructed from (real, imag)
                if (i == j) identity_matrix(i,j) = T(1.0, 0.0);
                else        identity_matrix(i,j) = T(0.0, 0.0);
            }
        }
        return identity_matrix;
    }
    
    // Sets to all zeros
    __host__ __device__ inline void setZero() {
        #pragma unroll
        for (int i=0; i < N*N; i++) data[i] = T(0.0 , 0.0);
    }

    // Sets to Identity matrix
    __host__ __device__ inline void setIdentity() {
    #pragma unroll
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N; j++) {
            (*this)(i, j) = (i == j) ? T(1.0) : T(0.0);
        }
    }
}

    __host__ __device__ constexpr int rows() const { return  N; }
    __host__ __device__ constexpr int columns() const { return  N; }
    __host__ __device__ constexpr int size() const { return N * N; }
    // 3. Accessors (Keep it simple)
    __host__ __device__ inline T& operator()(int i, int j) { return data[i * N + j]; }
    __host__ __device__ inline const T& operator()(int i, int j) const { return data[i * N + j]; }

    // 4. Core Physics Math
    // Conjugate Transpose (Dagger)
    __host__ __device__ inline Matrix<T, N> dagger() const {
        Matrix<T, N> res;
        #pragma unroll
        for (int i=0; i<N; i++) {
            #pragma unroll
            for (int j=0; j<N; j++) {
                res(i, j) = conj((*this)(j, i)); // Note the swap of i and j
            }
        }
        return res;
    }

    // Real part
    __host__ __device__ inline Matrix<T, N> real() const {
        Matrix<T, N> rel;
        #pragma unroll
        for (int i=0; i<N; i++) {
            #pragma unroll
            for (int j=0; j<N; j++) {
                rel(i, j) = T((*this)(i, j).real(), 0.0);
            }
        }
        return rel;
    }

    // Imaginary part
    __host__ __device__ inline Matrix<T, N> imag() const {
        Matrix<T, N> rel;
        #pragma unroll
        for (int i=0; i<N; i++) {
            #pragma unroll
            for (int j=0; j<N; j++) {
                rel(i, j) = T( (*this)(i, j).imag(),0.0);
            }
        }
        return rel;
    }


    // Trace (Used in Action calculation)
    __host__ __device__ inline T trace() const {
        T tr = T(0.0);
        #pragma unroll
        for (int i=0; i<N; i++) tr = tr+(*this)(i, i);
        return tr;
    }

    
    
};

//conj
template<typename T, int N>   
__device__ __host__ inline Matrix<T,N> conj(const Matrix<T,N> & a)
{
    Matrix<T,N> result;
#pragma unroll
    for (int i = 0; i < a.size(); i++) result.data[i] = conj(a.data[i]);
    return result;
}


// add
template<typename T, int N>   
__device__ __host__ inline Matrix<T,N> operator+(const Matrix<T,N> & a, const Matrix<T,N> & b)
{
    Matrix<T,N> result;
#pragma unroll
    for (int i = 0; i < a.size(); i++) result.data[i] = a.data[i] + b.data[i];
    return result;
}

template<typename T, int N>   
__device__ __host__ inline Matrix<T,N> operator-(const Matrix<T,N> & a, const Matrix<T,N> & b)
{
    Matrix<T,N> result;
#pragma unroll
    for (int i = 0; i < a.size(); i++) result.data[i] = a.data[i] - b.data[i];
    return result;
}


template<typename T, int N>   
__device__ __host__ inline Matrix<T,N> operator+= (Matrix<T,N> & a, const Matrix<T,N> & b)
{
#pragma unroll
    for (int i = 0; i < a.size(); i++) a.data[i] = a.data[i] + b.data[i];
    return a; 
}
template<typename T, int N>   
__device__ __host__ inline Matrix<T,N> operator-= (Matrix<T,N> & a, const Matrix<T,N> & b)
{
#pragma unroll
    for (int i = 0; i < a.size(); i++) a.data[i] = a.data[i] - b.data[i];
    return a; 
}

// C-multiply
template<typename T, int N>   
__device__ __host__ inline Matrix<T,N> operator*(const Matrix<T,N> & a, const T & b)
{
    Matrix<T,N> result;
#pragma unroll
    for (int i = 0; i < a.size(); i++) result.data[i] = a.data[i] * b;
    return result;
}
template<typename T, int N>   
__device__ __host__ inline Matrix<T,N> operator*(const T & a, const Matrix<T,N> & b)
{
    Matrix<T,N> result;
#pragma unroll
    for (int i = 0; i < b.size(); i++) result.data[i] = a* b.data[i];
    return result;
}

template<typename T, int N>   
__device__ __host__ inline Matrix<T,N> operator*(const Matrix<complex< T>,N> & a, const T & b)
{
    Matrix<T,N> result;
#pragma unroll
    for (int i = 0; i < a.size(); i++) result.data[i] = a.data[i] * b;
    return result;
}
template<typename T, int N>   
__device__ __host__ inline Matrix<T,N> operator*(const T & a, const Matrix<complex<T>,N> & b)
{
    Matrix<T,N> result;
#pragma unroll
    for (int i = 0; i < b.size(); i++) result.data[i] = a* b.data[i];
    return result;
}


template<typename T, int N>   
__device__ __host__ inline Matrix<T,N>& operator*=(Matrix<T,N> & a, const T & b)
{
#pragma unroll
    for (int i = 0; i < a.size(); i++) a.data[i] = a.data[i] * b;
    return a;
}

template<typename T, int N>   
__device__ __host__ inline Matrix<T,N>& operator*=( const T & a,Matrix<T,N> & b)
{
#pragma unroll
    for (int i = 0; i < b.size(); i++) b.data[i] = b.data[i] * a;
    return b;
}

template<typename T, int N>   
__device__ __host__ inline Matrix<T,N>& operator*=(Matrix<complex<T>,N> & a, const T & b)
{
#pragma unroll
    for (int i = 0; i < a.size(); i++) a.data[i] = a.data[i] * b;
    return a;
}

template<typename T, int N>   
__device__ __host__ inline Matrix<T,N>& operator*=( const T & a,Matrix<complex<T>,N> & b)
{
#pragma unroll
    for (int i = 0; i < b.size(); i++) b.data[i] = b.data[i] * a;
    return b;
}

// not inherit from matrix class, but direct inside the matrix class (in this project we will only deal with 3*3 matrix including SU3 and P momentum matrix)

template<typename T, int N>
__device__ __host__ inline Matrix<T,N> operator* (const Matrix<T,N>& A, const Matrix<T, N>& B) {
    Matrix<T, N> C;
    C.setZero(); 
    #pragma unroll
    for (int i = 0; i < N; ++i) { 
        #pragma unroll
        for (int j = 0; j < N; ++j) { 
            T sum = T(0.0);
            #pragma unroll
            for (int k = 0; k < N; ++k) { 
                sum = sum + (A(i, k) * B(k, j));
            }
            C(i, j) = sum;
        }
    }
    return C;
}

template<typename T>
__device__ __host__ inline T Det(const Matrix<T,3>& A){
    T term1 = A(0,0) * (A(1,1) * A(2,2) - A(1,2) * A(2,1));
    T term2 = A(0,1) * (A(1,0) * A(2,2) - A(1,2) * A(2,0));
    T term3 = A(0,2) * (A(1,0) * A(2,1) - A(1,1) * A(2,0));
    return term1 - term2 + term3;
}

// =============================================================================
// SU(3) Generators: T^a = λ^a / 2 (a = 1, ..., 8)
// These are used for projecting forces onto the Lie algebra
// =============================================================================

namespace su3_generators {

template<typename T>
__device__ __host__ inline Matrix<complex<T>, 3> generator(int a) {
    Matrix<complex<T>, 3> Tmat;
    Tmat.setZero();
    
    switch(a) {
        case 0:  // λ^1
            Tmat(0, 1) = complex<T>(0, 1);
            Tmat(1, 0) = complex<T>(0, 1);
            break;
        case 1:  // λ^2
            Tmat(0, 1) = complex<T>(-1, 0);
            Tmat(1, 0) = complex<T>(1, 0);
            break;
        case 2:  // λ^3
            Tmat(0, 0) = complex<T>(1, 0);
            Tmat(1, 1) = complex<T>(-1, 0);
            break;
        case 3:  // λ^4
            Tmat(0, 2) = complex<T>(0, 1);
            Tmat(2, 0) = complex<T>(0, 1);
            break;
        case 4:  // λ^5
            Tmat(0, 2) = complex<T>(-1, 0);
            Tmat(2, 0) = complex<T>(1, 0);
            break;
        case 5:  // λ^6
            Tmat(1, 2) = complex<T>(0, 1);
            Tmat(2, 1) = complex<T>(0, 1);
            break;
        case 6:  // λ^7
            Tmat(1, 2) = complex<T>(-1, 0);
            Tmat(2, 1) = complex<T>(1, 0);
            break;
        case 7:  // λ^8 (diagonal)
            Tmat(0, 0) = complex<T>(1, 0);
            Tmat(1, 1) = complex<T>(1, 0);
            Tmat(2, 2) = complex<T>(-2, 0);
            break;
    }
    Tmat *= complex<T>(T(0.5), 0);
    return Tmat;
}

template<typename T>
__device__ __host__ inline T trace_generator_product(int a, int b) {
    return (a == b) ? T(0.5) : T(0);
}

} // namespace su3_generators

} // namespace qcdcuda