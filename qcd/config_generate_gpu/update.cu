// vector_add.cu

#include <cuda_runtime.h>
#include <cstdio>

// Macro for checking CUDA error codes
#define CHECK(call)                                                          \
{                                                                            \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess)                                                \
    {                                                                        \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);               \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                                             \
    }                                                                        \
}

// ----------------------------------------------------
// The CUDA Kernel: Executes in parallel on the GPU
// ----------------------------------------------------
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    // Calculate the 1D global index for this thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds to ensure the index is within the vector size
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

// ----------------------------------------------------
// Host Wrapper Function: Manages memory and launches the kernel
// ----------------------------------------------------
extern "C" void launch_vector_add(const float *h_A, const float *h_B, float *h_C, int N)
{
    // --- 1. Define grid and block configuration ---
    const int threadsPerBlock = 256;
    // Calculate the number of blocks needed to cover all N elements
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // --- 2. Allocate Device Memory (d_A, d_B, d_C) ---
    float *d_A, *d_B, *d_C;
    size_t bytes = N * sizeof(float);

    CHECK(cudaMalloc((void**)&d_A, bytes));
    CHECK(cudaMalloc((void**)&d_B, bytes));
    CHECK(cudaMalloc((void**)&d_C, bytes));

    // --- 3. Copy Host data (h_A, h_B) to Device (d_A, d_B) ---
    CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // --- 4. Launch the Kernel ---
    printf("Launching kernel with %d blocks and %d threads per block.\n", blocksPerGrid, threadsPerBlock);
    
    // The triple chevrons <<<...>>> specify the grid and block configuration
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Synchronize to ensure the kernel has finished before proceeding
    CHECK(cudaDeviceSynchronize());

    // --- 5. Copy Device result (d_C) back to Host (h_C) ---
    CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // --- 6. Clean up Device Memory ---
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
}