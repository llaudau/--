#include "matrix.cuh"
#include "complex.cuh"
#include "gauge_operation.cuh"
#include <cuda_runtime.h>

using namespace qcdcuda;

// The actual GPU kernel
__global__ void addMatricesKernel(Matrix<complex<float>, 3>* d_A, 
                                  Matrix<complex<float>, 3>* d_B, 
                                  Matrix<complex<float>, 3>* d_C) {
    // Basic test: just thread 0 does the work
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_C = *d_A + *d_B;
    }
}
__global__ void addmnsKernel(Matrix<complex<float>, 3>* d_A, 
                                  complex<float> *d_B, 
                                  Matrix<complex<float>, 3>* d_C) {
    // Basic test: just thread 0 does the work
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_C = *d_A * *d_B;
    }

}


// Wrapper function that the .cpp file will call
extern "C" void launch_gpu_add(void* h_A, void* h_B, void* h_C) {
    Matrix<complex<float>, 3> *d_A, *d_B, *d_C;

    // Allocate
    cudaMalloc(&d_A, sizeof(Matrix<complex<float>, 3>));
    cudaMalloc(&d_B, sizeof(Matrix<complex<float>, 3>));
    cudaMalloc(&d_C, sizeof(Matrix<complex<float>, 3>));

    // Copy to Device
    cudaMemcpy(d_A, h_A, sizeof(Matrix<complex<float>, 3>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(Matrix<complex<float>, 3>), cudaMemcpyHostToDevice);

    // Launch
    addMatricesKernel<<<1, 1>>>(d_A, d_B, d_C);

    // Copy back to Host
    cudaMemcpy(h_C, d_C, sizeof(Matrix<complex<float>, 3>), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}



extern "C" void launch_gpu_addmn(void* h_A, void* h_B, void* h_C) {
    Matrix<complex<float>, 3> *d_A, *d_C;
    complex<float> *d_B;
    // Allocate
    cudaMalloc(&d_A, sizeof(Matrix<complex<float>, 3>));
    cudaMalloc(&d_B, sizeof(complex<float>));
    cudaMalloc(&d_C, sizeof(Matrix<complex<float>, 3>));

    // Copy to Device
    cudaMemcpy(d_A, h_A, sizeof(Matrix<complex<float>, 3>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(complex<float>), cudaMemcpyHostToDevice);

    // Launch
    
    addmnsKernel<<<1, 1>>>(d_A, d_B, d_C);
    
    // Copy back to Host
    cudaMemcpy(h_C, d_C, sizeof(Matrix<complex<float>, 3>), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}