#include "lattice.cuh"
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

//kernels 
template <typename num_type>
__global__ void run_randomize_kernel(LatticeView lat, num_type epsilon, unsigned long seed) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lat.volume) return;

    curandState state;
    curand_init(seed, site, 0, &state);


    Matrix<complex<num_type>, 3> U;
    for (int mu=0;mu<4;mu++){
        generate_random_su3_near_identity(U, epsilon, &state);
        lat.d_links[lat.link_idx(site,mu)] = U;

        // check reunitarize  
        // if (site==0 and mu==0){
        //    complex<num_type> detU=Det(U);
        //    Matrix<complex <num_type>,3> UU=U*U.dagger();
        //    printf("Det(U) at site 0, mu 0: Real = %f, Imag = %f\n", (double) (detU.real()), (double)(detU.imag()));
        //    printf("UU(0,0) at site 0, mu 0: Real = %f, Imag = %f\n", (double) (UU(0,0).real()), (double)(UU(0,0).imag()));
        //    printf("UU(1,1) at site 0, mu 0: Real = %f, Imag = %f\n", (double) (UU(1,2).real()), (double)(UU(1,2).imag()));
        //    printf("UU(2,2) at site 0, mu 0: Real = %f, Imag = %f\n", (double) (UU(2,2).real()), (double)(UU(2,2).imag()));
        // }
    }
}


__global__ void setup_rng_kernel(curandState* states, unsigned long long seed, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        /* Each thread gets same seed, but a different sequence number.
           This ensures no overlap between threads. */
        curand_init(seed, id, 0, &states[id]);
    }
}
//functions

void GaugeField::init_hopping() {
    // 1. Create a temporary host buffer
    // volume * 8 (4 directions * 2 for Up/Down)
    std::vector<int> h_hopping(params.volume * 8);
    for (int t = 0; t < params.L[3]; t++) {
    for (int z = 0; z < params.L[2]; z++) {
    for (int y = 0; y < params.L[1]; y++) {
    for (int x = 0; x < params.L[0]; x++) {
        
        int curr = (((t * params.L[2] + z) * params.L[1] + y) * params.L[0] + x);

        // Pre-caculate shifted coordinates with Periodic Boundaries
        int xp = (x + 1) % params.L[0];
        int xm = (x - 1 + params.L[0]) % params.L[0];
        int yp = (y + 1) % params.L[1];
        int ym = (y - 1 + params.L[1]) % params.L[1];
        int zp = (z + 1) % params.L[2];
        int zm = (z - 1 + params.L[2]) % params.L[2];
        int tp = (t + 1) % params.L[3];
        int tm = (t - 1 + params.L[3]) % params.L[3];

        // Store indices in the buffer
        h_hopping[curr  + 0*params.volume] = (((t * params.L[2] + z) * params.L[1] + y) * params.L[0] + xp); 
        h_hopping[curr  + 4*params.volume] = (((t * params.L[2] + z) * params.L[1] + y) * params.L[0] + xm); 

        h_hopping[curr  + 1*params.volume] = (((t * params.L[2] + z) * params.L[1] + yp) * params.L[0] + x); 
        h_hopping[curr  + 5*params.volume] = (((t * params.L[2] + z) * params.L[1] + ym) * params.L[0] + x); 

        h_hopping[curr  + 2*params.volume] = (((t * params.L[2] + zp) * params.L[1] + y) * params.L[0] + x); 
        h_hopping[curr  + 6*params.volume] = (((t * params.L[2] + zm) * params.L[1] + y) * params.L[0] + x); 

        h_hopping[curr  + 3*params.volume] = (((tp * params.L[2] + z) * params.L[1] + y) * params.L[0] + x); 
        h_hopping[curr  + 7*params.volume] = (((tm * params.L[2] + z) * params.L[1] + y) * params.L[0] + x); 

    }}}}

    // 2. Copy to GPU
    cudaMemcpy(d_hopping, h_hopping.data(), params.volume * 8 * sizeof(int), cudaMemcpyHostToDevice);
}


bool GaugeField::acc_rej(num_type Hi, num_type Hf) {
    num_type r = dist(engine);

    // std::cout<<"Hi : "<<Hi<<std::endl;
    // std::cout<<"Hf : "<<Hf<<std::endl;
    dH=Hf-Hi;
    // std::cout<<"Hi : "<<Hi<<std::endl;
    // std::cout<<"Hf : "<<Hf<<std::endl;
    // std::cout<<"dH : "<<dH<<std::endl;
    if (std::exp(Hi-Hf) > r) {
        return true;
    } 
    return false;
}


void GaugeField::init_links(num_type epsi, unsigned long seed){
    run_randomize_kernel<<<params.blocks,params.threads>>>(this->view(), epsi, seed);
}

void GaugeField::init_rng(unsigned long long seed){
    setup_rng_kernel<<<params.blocks,params.threads>>>(d_rng_states,seed,params.volume);
}

num_type GaugeField::calculate_plaquette() {
    kernel_calculate_plaquette<<<params.blocks, params.threads>>>(this->view(), d_workspace);
    thrust::device_ptr<num_type> ptr(d_workspace);
    num_type avg_plaq = thrust::reduce(thrust::device, ptr, ptr + params.volume, (num_type)0.0, thrust::plus<num_type>());
    return avg_plaq / params.volume;
}