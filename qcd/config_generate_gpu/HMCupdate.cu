#include "lattice.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdexcept>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
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


// HMC update kernel

__global__ void random_momentum(
    double* mom,
    curandState* rng,
    int Ns, int Nt 
) {
    int v= blockIdx.x *blockDim.x +threadIdx.x;
    int volume =Ns*Ns*Ns*Nt;
    if (v>=volume) return;

    curandState localState = rng[v]; // Load
    mom[v] = curand_normal_double(&localState);
    rng[v] = localState; // Save back!
}

__global__ void update_field(
    double* field ,
    double* momentum ,
    int Ns, int Nt,
    double epsi
) {
    int v= blockIdx.x *blockDim.x +threadIdx.x;
    int volume =Ns*Ns*Ns*Nt;
    if (v>=volume) return;
    field[v]+=epsi*momentum[v];

}

__global__ void update_momentum(
    double* field ,
    double* momentum ,
    int *hopping_data,
    int Ns, int Nt,
    double epsi, double lambda,double beta
) {
    int v= blockIdx.x *blockDim.x +threadIdx.x;
    int volume =Ns*Ns*Ns*Nt;
    if (v>=volume) return;
    double phin=0.0;
    for (int mu=0;mu<8;mu++) phin+=field[hopping_data[v+mu*volume]];
    double force = 2.0 * beta * phin - 4.0 * lambda * (field[v]*field[v] - 1.0) * field[v] - 2.0 * field[v];
    // momentum update: p <- p + eps * force  (force is -dS/dphi)
    momentum[v] += epsi * force;
    
}

// Hamiltonian 

__global__ void Hamiltoniaper_site(
    double* energy_array, 
    double* field, 
    double* mom, 
    int * hopping_field,
    double beta, 
    double lambda, 
    int Ns, int Nt
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int volume = Ns * Ns * Ns * Nt;
    if (v >= volume) return;
    double h_v=0.5*mom[v]*mom[v];
    double phi2=field[v]*field[v];
    double phin=0;
    for (int mu=0 ;mu<4 ;mu++ ) phin+=field[hopping_field[v+2*mu*volume]];
    double action=-2*beta*phin*field[v]+phi2+lambda*(phi2-1.0)*(phi2-1.0);
    h_v+=action;
    energy_array[v]=h_v;
}

/* ============================================================
   Lattice CUDA methods
   ============================================================ */

void Lattice::cuda_init(unsigned long seed)
{
    size_t bytes = volume * sizeof(double);
    size_t bytes_hopping= volume *8*sizeof(int);
    // Allocate device field
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&m_data, bytes);
    cudaMalloc(&d_old_data, bytes);
    cudaMalloc(&d_energy_array, bytes);


    cudaMalloc(&hopping_data, bytes_hopping);
    cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(m_data, mom, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(hopping_data, hopping_field, bytes_hopping, cudaMemcpyHostToDevice);

    // Allocate RNG states
    cudaMalloc(&d_rng, volume * sizeof(curandState));
    init_rng_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_rng, volume, seed
    );

    cudaDeviceSynchronize();
}

void Lattice::cuda_finalize()
{
    size_t bytes = volume * sizeof(double);

    cudaMemcpy(data, d_data, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_rng);
    cudaFree(hopping_data);
    cudaFree(d_energy_array);

    d_data = nullptr;
    d_rng  = nullptr;
    hopping_data =nullptr;
    d_energy_array= nullptr;
}

bool Lattice::cuda_update_trajectory(double epsi, int num)
{   
    random_momentum<<<blocksPerGrid, threadsPerBlock>>>(m_data,d_rng,Ns,Nt);
    double H_old=Hamiltonian(d_data,m_data,Ns,Nt);

    cudaMemcpy(d_old_data, d_data, volume * sizeof(double), cudaMemcpyDeviceToDevice);

    // update_field<<<blocksPerGrid, threadsPerBlock>>>(d_data,m_data, Ns,Nt,epsi/2.0);
    for (int i=0; i<num; i++){
        update_field<<<blocksPerGrid, threadsPerBlock>>>(d_data,m_data, Ns,Nt,epsi/2.0);
        update_momentum<<<blocksPerGrid, threadsPerBlock>>>(d_data,m_data,hopping_data, Ns,Nt,epsi,lambda,beta);
        // if (i<num-1){
        update_field<<<blocksPerGrid, threadsPerBlock>>>(d_data,m_data, Ns,Nt,epsi/2.0);
        // }
    }
    // update_field<<<blocksPerGrid, threadsPerBlock>>>(d_data,m_data, Ns,Nt,epsi/2.0);

    double H_new = Hamiltonian(d_data, m_data, Ns, Nt);

    double r = (double)rand() / RAND_MAX; // Simple CPU random for the test
    double dH = H_new - H_old;
    // std::cout<<(H_new)<<std::endl;
    expdh+=exp(-dH);
    if (r < exp(-dH)) {
        return true; 
    } 
    else {
        cudaMemcpy(d_data, d_old_data, volume * sizeof(double), cudaMemcpyDeviceToDevice);
        return false;
    }
}

double Lattice::Hamiltonian (double* d_field, double* d_mom, int Ns, int Nt) {
    int volume = Ns * Ns * Ns * Nt;


    // Launch Kernel
    Hamiltoniaper_site<<<blocksPerGrid, threadsPerBlock>>>(d_energy_array, d_field,d_mom,hopping_data,beta,lambda,Ns,Nt);

    // Reduction
    thrust::device_ptr<double> ptr(d_energy_array);
    // Add 0.0 to ensure the reduction uses double precision and starts at zero
    double total_H = thrust::reduce(ptr, ptr + volume, 0.0, thrust::plus<double>());

    return total_H;
}
