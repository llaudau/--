#pragma once

#include <cuda_runtime.h>

namespace qcdcuda
{
// in this file defining the fma : a*b+c; and mul2 a*b; and add2 a+b; in the complex.cuh we will use this to define the complex add,div and mul.
  
  inline __host__ __device__ float abs(const float a) { return fabs(a); }
  inline __host__ __device__ double abs(const double a) { return fabs(a); }

__device__ __host__ inline float2 fma2(float2 a, float2 b, float2 c) { return {a.x * b.x + c.x, a.y * b.y + c.y}; }
  __device__ __host__ inline double2 fma2(double2 a, double2 b, double2 c)
  {
    return {a.x * b.x + c.x, a.y * b.y + c.y};
  }

  __device__ __host__ inline float2 mul2(float2 a, float2 b) { return {a.x * b.x, a.y * b.y}; }
  __device__ __host__ inline double2 mul2(double2 a, double2 b) { return {a.x * b.x, a.y * b.y}; }

  __device__ __host__ inline float2 add2(float2 a, float2 b) { return {a.x + b.x, a.y + b.y}; }
  __device__ __host__ inline double2 add2(double2 a, double2 b) { return {a.x + b.x, a.y + b.y}; }


  // fast sqrt 
  template <typename T>
    __device__ __forceinline__ T gpu_rsqrt(T x);

    // Specialization for float
    template <>
    __device__ __forceinline__ float gpu_rsqrt<float>(float x) {
        return rsqrtf(x);
    }

    // Specialization for double
    template <>
    __device__ __forceinline__ double gpu_rsqrt<double>(double x) {
        return rsqrt(x);
    }
}
