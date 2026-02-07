#pragma once
#include <cuda_runtime.h>
#include <random>
#include <cmath>
#include <vector>
#include "gauge_operation.cuh" // Your P and U operations
using namespace qcdcuda;
using num_type=float;
struct LatticeParams {
    int L[4]; // Dimensions: x, y, z, t
    int volume;
    int threads=256;
    num_type beta;
    int blocks; 
};

struct LatticeView {
    Matrix<complex<num_type>, 3>* d_links; 
    Matrix<complex<num_type>,3>* d_moms;
    int* d_hopping;                 
    int volume;
    int Lx, Ly, Lz, Lt;
    num_type beta;
    curandState * d_rng;

    // This is where you put your indexing logic
    __device__ inline int link_idx(int site, int mu) const {
        return site + mu * volume;
    }

    // when calculating H, i each thread-> a site, need this to get the index
    __device__ inline int neighbor(int site, int mu_dir) const {
        // mu_dir: 0-3 for UP, 4-7 for DOWN
        return d_hopping[mu_dir * volume + site];
    }
    //when calculating update, each thread -> a link 
};



class GaugeField {

private:

    // the acceptor of Hamiltonian is here
    std::mt19937 engine; 
    std::uniform_real_distribution<num_type> dist;
public:
    


    Matrix<complex<num_type>,3>* d_links; // Pointer to GPU memory
    Matrix<complex<num_type>,3>* d_links_old;
    Matrix<complex<num_type>,3>* d_links_smeared; // Buffer for smearing
    Matrix<complex<num_type>,3>* d_moms;
    curandState* d_rng_states;

    num_type* d_workspace;
    int* d_hopping;
    LatticeParams params;
    num_type dH=0;
    // i need a random number to help me calculate the update


    GaugeField(int Lx, int Ly, int Lz, int Lt,num_type Beta) {
        params.L[0] = Lx; params.L[1] = Ly; params.L[2] = Lz; params.L[3] = Lt;
        params.volume = Lx * Ly * Lz * Lt;
        params.beta=Beta;
        params.blocks= (params.volume+ params.threads - 1) / params.threads;
        unsigned long long seed = 1234ULL;
        unsigned long long seed_links = 4321ULL;
        // Each site has 4 directions (links)
        size_t size = params.volume * 4 * sizeof(Matrix<complex<num_type>,3>);
        size_t size_hopping = params.volume*8 * sizeof(int);
        size_t size_rng= params.volume *sizeof(curandState);
        size_t size_density= params.volume * sizeof (num_type);
        cudaMalloc(&d_links, size);
        cudaMalloc(&d_links_old, size);
        cudaMalloc(&d_links_smeared, size);
        cudaMalloc(&d_workspace,2*size_density);
        cudaMalloc(&d_moms, size);
        cudaMalloc(&d_hopping,size_hopping);
        cudaMalloc(&d_rng_states,size_rng);
        init_hopping();
        init_links(0.5f,seed_links);
        init_rng(seed);
    }
    ~GaugeField() {
        cudaFree(d_links);
        cudaFree(d_links_old);
        cudaFree(d_links_smeared);
        cudaFree(d_moms);
        cudaFree(d_hopping);
    }
    LatticeView view() {
        LatticeView v;
        v.d_links   = this->d_links;
        v.d_moms= this->d_moms;
        v.d_hopping = this->d_hopping;
        v.d_rng=this->d_rng_states;

        v.volume    = this->params.volume;
        v.Lx = params.L[0]; v.Ly = params.L[1]; 
        v.Lz = params.L[2]; v.Lt = params.L[3];
        v.beta=this->params.beta;
        return v;
    }
    //initialize hopping and links
    void init_hopping();
    void init_links(num_type epsi, unsigned long seed);
    void init_rng(unsigned long long seed);

    bool acc_rej(num_type H_i, num_type H_f);

    // Phys obsev 
    num_type Hamilt(num_type &out_action);
    num_type topo_charge();
    num_type calculate_plaquette();

    //HMC update
    void update_1step(num_type length, int num_steps);

    //Metropolis update
    num_type metropolis_update(num_type epsilon, int nsteps);

    //Smearing
    void smear_links(num_type alpha, int n_iter);

};

__global__ void kernel_calculate_plaquette(LatticeView lat, num_type* results);
__global__ void kernel_metropolis(LatticeView lat, num_type epsilon, curandState* d_states, int* d_accept);
__global__ void kernel_smear_links(LatticeView lat, Matrix<complex<num_type>, 3>* d_links_new, num_type alpha);
__global__ void kernel_reunitarize_links(LatticeView lat);

void gauge_field_smear(Matrix<complex<num_type>, 3>* d_links, Matrix<complex<num_type>, 3>* d_links_new, 
                        int* d_hopping, int volume, int blocks, int threads, num_type alpha);