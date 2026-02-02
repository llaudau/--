#include "lattice.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdexcept>

/* ============================================================
   CUDA kernels
   ============================================================ */

// Initialize RNG states (one per lattice site)
__global__
void init_rng_kernel(
    curandState* rng,
    int volume,
    unsigned long seed
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= volume) return;

    // One independent RNG stream per site
    curand_init(seed, v, 0, &rng[v]);
}


// Checkerboard update kernel

__global__
void update_checkerboard_kernel(
    double* data,
    curandState* rng,
    int Ns, int Nt,
    int parity
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    int volume = Ns * Ns * Ns * Nt;
    if (v >= volume) return;

    // Decode coordinates
    int x = v % Ns;
    int y = (v / Ns) % Ns;
    int z = (v / (Ns * Ns)) % Ns;
    int t = v / (Ns * Ns * Ns);

    // Checkerboard condition
    if (((x + y + z + t) & 1) != parity) return;

    // Draw random number and update field
    double r = curand_uniform_double(&rng[v]);
    data[v] = r;
}



/* ============================================================
   Lattice CUDA methods
   ============================================================ */

void Lattice::cuda_init(unsigned long seed)
{
    size_t bytes = volume * sizeof(double);

    // Allocate device field
    cudaMalloc(&d_data, bytes);

    cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice);

    // Allocate RNG states
    cudaMalloc(&d_rng, volume * sizeof(curandState));

    int threads = 256;
    int blocks  = (volume + threads - 1) / threads;

    init_rng_kernel<<<blocks, threads>>>(
        d_rng, volume, seed
    );

    cudaDeviceSynchronize();
}



void Lattice::cuda_update_sweep()
{
    int threads = 256;
    int blocks  = (volume + threads - 1) / threads;

    // Even sites
    update_checkerboard_kernel<<<blocks, threads>>>(
        d_data, d_rng, Ns, Nt, 0
    );

    // Odd sites
    update_checkerboard_kernel<<<blocks, threads>>>(
        d_data, d_rng, Ns, Nt, 1
    );
}














void Lattice::cuda_finalize()
{
    size_t bytes = volume * sizeof(double);

    cudaMemcpy(data, d_data, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_rng);

    d_data = nullptr;
    d_rng  = nullptr;
}



