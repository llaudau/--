#include "lattice.cuh"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <cmath>

namespace qcdcuda {

template<typename T>
__global__ void kernel_cg_axpy(FermionFieldView y, complex<T> a, FermionFieldView x, int volume) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if(site >= volume) return;
    
    #pragma unroll
    for(int c = 0; c < 3; c++) {
        #pragma unroll
        for(int s = 0; s < 4; s++) {
            y(site)(c, s) += a * x(site)(c, s);
        }
    }
}

template<typename T>
__global__ void kernel_cg_scale(FermionFieldView x, complex<T> a, int volume) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if(site >= volume) return;
    
    #pragma unroll
    for(int c = 0; c < 3; c++) {
        #pragma unroll
        for(int s = 0; s < 4; s++) {
            x(site)(c, s) *= a;
        }
    }
}

template<typename T>
__global__ void kernel_cg_zero(FermionFieldView x, int volume) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if(site >= volume) return;
    
    x(site).setZero();
}

template<typename T>
__global__ void kernel_spinor_reduce_sum(FermionFieldView a, FermionFieldView b, T* result, int volume) {
    extern __shared__ T sdata[];
    
    int tid = threadIdx.x;
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    
    T local_sum = T(0);
    if(site < volume) {
        // Manually compute spinor inner product: Σ_c,s conj(ψ₁(c,s)) × ψ₂(c,s)
        for(int c = 0; c < 3; c++) {
            for(int s = 0; s < 4; s++) {
                auto psi1_comp = a(site)(c, s);  // complex<T>
                auto psi2_comp = b(site)(c, s);  // complex<T>
                // conj(psi1) * psi2, then extract real part
                local_sum += (conj(psi1_comp) * psi2_comp).real();
            }
        }
    }
    
    sdata[tid] = local_sum;
    __syncthreads();
    
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if(tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

template<typename T>
T reduce_spinor_dot(FermionFieldView a, FermionFieldView b, int volume, int blocks, int threads) {
    T* d_result;
    cudaMalloc(&d_result, blocks * sizeof(T));
    
    kernel_spinor_reduce_sum<T><<<blocks, threads, threads * sizeof(T)>>>(a, b, d_result, volume);
    cudaDeviceSynchronize();
    
    thrust::device_ptr<T> ptr(d_result);
    T sum = thrust::reduce(ptr, ptr + blocks, T(0), thrust::plus<T>());
    
    cudaFree(d_result);
    return sum;
}

template<typename T>
CGResult solve_cg(FermionFieldView b, FermionFieldView x, LatticeView lat, 
                  int max_iter, T tolerance) {
    CGResult result;
    result.converged = false;
    result.iterations = 0;
    
    int threads = 256;
    int blocks = (lat.volume + threads - 1) / threads;
    
    FermionFieldView r;
    r.d_psi = nullptr;
    cudaMalloc(&r.d_psi, lat.volume * sizeof(Spinor<complex<T>>));
    r.volume = lat.volume;
    
    FermionFieldView p;
    p.d_psi = nullptr;
    cudaMalloc(&p.d_psi, lat.volume * sizeof(Spinor<complex<T>>));
    p.volume = lat.volume;
    
    FermionFieldView Ap;
    Ap.d_psi = nullptr;
    cudaMalloc(&Ap.d_psi, lat.volume * sizeof(Spinor<complex<T>>));
    Ap.volume = lat.volume;
    
    cudaMemcpy(r.d_psi, b.d_psi, lat.volume * sizeof(Spinor<complex<T>>), cudaMemcpyDeviceToDevice);
    kernel_cg_zero<T><<<blocks, threads>>>(x, lat.volume);
    kernel_cg_zero<T><<<blocks, threads>>>(p, lat.volume);
    cudaMemcpy(p.d_psi, r.d_psi, lat.volume * sizeof(Spinor<complex<T>>), cudaMemcpyDeviceToDevice);
    
    T rsold = reduce_spinor_dot<T>(r, r, lat.volume, blocks, threads);
    T rsnew;
    
    for(int iter = 0; iter < max_iter; iter++) {
        kernel_wilson_dslash<<<blocks, threads>>>(lat, p, Ap, 0);
        cudaDeviceSynchronize();
        
        T pAp = reduce_spinor_dot<T>(p, Ap, lat.volume, blocks, threads);
        
        if(pAp == T(0)) {
            result.iterations = iter;
            result.residual = sqrt(rsold);
            cudaFree(r.d_psi);
            cudaFree(p.d_psi);
            cudaFree(Ap.d_psi);
            return result;
        }
        
        T alpha = rsold / pAp;
        
        kernel_cg_axpy<T><<<blocks, threads>>>(x, complex<T>(alpha, 0), p, lat.volume);
        kernel_cg_axpy<T><<<blocks, threads>>>(r, complex<T>(-alpha, 0), Ap, lat.volume);
        cudaDeviceSynchronize();
        
        rsnew = reduce_spinor_dot<T>(r, r, lat.volume, blocks, threads);
        
        if(sqrt(rsnew) < tolerance) {
            result.converged = true;
            result.iterations = iter + 1;
            result.residual = sqrt(rsnew);
            cudaFree(r.d_psi);
            cudaFree(p.d_psi);
            cudaFree(Ap.d_psi);
            return result;
        }
        
        T beta = rsnew / rsold;
        
        kernel_cg_scale<T><<<blocks, threads>>>(p, complex<T>(beta, 0), lat.volume);
        kernel_cg_axpy<T><<<blocks, threads>>>(p, complex<T>(1, 0), r, lat.volume);
        cudaDeviceSynchronize();
        
        rsold = rsnew;
    }
    
    result.iterations = max_iter;
    result.residual = sqrt(rsold);
    cudaFree(r.d_psi);
    cudaFree(p.d_psi);
    cudaFree(Ap.d_psi);
    return result;
}

template<typename T>
CGResult solve_cg_mdaggerm(FermionFieldView b, FermionFieldView x, LatticeView lat,
                           int max_iter, T tolerance) {
    CGResult result;
    result.converged = false;
    result.iterations = 0;
    
    int threads = 256;
    int blocks = (lat.volume + threads - 1) / threads;
    
    FermionFieldView r;
    r.d_psi = nullptr;
    cudaMalloc(&r.d_psi, lat.volume * sizeof(Spinor<complex<T>>));
    r.volume = lat.volume;
    
    FermionFieldView p;
    p.d_psi = nullptr;
    cudaMalloc(&p.d_psi, lat.volume * sizeof(Spinor<complex<T>>));
    p.volume = lat.volume;
    
    FermionFieldView Mp;
    Mp.d_psi = nullptr;
    cudaMalloc(&Mp.d_psi, lat.volume * sizeof(Spinor<complex<T>>));
    Mp.volume = lat.volume;
    
    FermionFieldView tmp;
    tmp.d_psi = nullptr;
    cudaMalloc(&tmp.d_psi, lat.volume * sizeof(Spinor<complex<T>>));
    tmp.volume = lat.volume;
    
    cudaMemcpy(r.d_psi, b.d_psi, lat.volume * sizeof(Spinor<complex<T>>), cudaMemcpyDeviceToDevice);
    kernel_cg_zero<T><<<blocks, threads>>>(x, lat.volume);
    kernel_cg_zero<T><<<blocks, threads>>>(p, lat.volume);
    cudaMemcpy(p.d_psi, r.d_psi, lat.volume * sizeof(Spinor<complex<T>>), cudaMemcpyDeviceToDevice);
    
    T rsold = reduce_spinor_dot<T>(r, r, lat.volume, blocks, threads);
    T rsnew;
    
    for(int iter = 0; iter < max_iter; iter++) {
        kernel_wilson_dslash<<<blocks, threads>>>(lat, p, tmp, 0);
        cudaDeviceSynchronize();
        kernel_wilson_dslash<<<blocks, threads>>>(lat, tmp, Mp, 1);
        cudaDeviceSynchronize();
        
        T pMp = reduce_spinor_dot<T>(p, Mp, lat.volume, blocks, threads);
        
        if(fabs(pMp) < std::numeric_limits<T>::epsilon()) {
            result.iterations = iter;
            result.residual = sqrt(rsold);
            cudaFree(r.d_psi);
            cudaFree(p.d_psi);
            cudaFree(Mp.d_psi);
            cudaFree(tmp.d_psi);
            return result;
        }
        
        T alpha = rsold / pMp;
        
        kernel_cg_axpy<T><<<blocks, threads>>>(x, complex<T>(alpha, 0), p, lat.volume);
        kernel_cg_axpy<T><<<blocks, threads>>>(r, complex<T>(-alpha, 0), Mp, lat.volume);
        cudaDeviceSynchronize();
        
        rsnew = reduce_spinor_dot<T>(r, r, lat.volume, blocks, threads);
        
        if(sqrt(rsnew) < tolerance) {
            result.converged = true;
            result.iterations = iter + 1;
            result.residual = sqrt(rsnew);
            cudaFree(r.d_psi);
            cudaFree(p.d_psi);
            cudaFree(Mp.d_psi);
            cudaFree(tmp.d_psi);
            return result;
        }
        
        T beta = rsnew / rsold;
        
        kernel_cg_scale<T><<<blocks, threads>>>(p, complex<T>(beta, 0), lat.volume);
        kernel_cg_axpy<T><<<blocks, threads>>>(p, complex<T>(1, 0), r, lat.volume);
        cudaDeviceSynchronize();
        
        rsold = rsnew;
    }
    
    result.iterations = max_iter;
    result.residual = sqrt(rsold);
    cudaFree(r.d_psi);
    cudaFree(p.d_psi);
    cudaFree(Mp.d_psi);
    cudaFree(tmp.d_psi);
    return result;
}

} // namespace qcdcuda

namespace qcdcuda {

template CGResult solve_cg<float>(FermionFieldView, FermionFieldView, LatticeView, int, float);
template CGResult solve_cg_mdaggerm<float>(FermionFieldView, FermionFieldView, LatticeView, int, float);

} // namespace qcdcuda
