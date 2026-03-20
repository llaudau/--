#pragma once
#include <cuda_runtime.h>
#include <random>
#include <cmath>
#include <vector>
#include "gauge_operation.cuh"
#include "dirac.cuh"
// using namespace qcdcuda;

using num_type=float;
namespace qcdcuda{
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
    num_type kappa;
    curandState * d_rng;

    // Gauge link indexing: site + mu * volume
    __device__ inline int link_idx(int site, int mu) const {
        return site + mu * volume;
    }

    // Neighbor lookup: mu_dir (0-3=UP, 4-7=DOWN)
    __device__ inline int neighbor(int site, int mu_dir) const {
        return d_hopping[mu_dir * volume + site];
    }
};

struct FermionFieldView {
    Spinor<complex<num_type>>* d_psi;
    int volume;

    // Spinor indexing: site * 12 + color * 4 + spin
    // Layout: [volume][3][4] - one full spinor per lattice site
    __device__ inline int spinor_idx(int site, int color, int spin) const {
        return site * 12 + color * 4 + spin;
    }

    __device__ inline int spinor_site_offset(int site) const {
        return site * 12;
    }

    __device__ inline Spinor<complex<num_type>>& operator()(int site) {
        return d_psi[site];
    }

    __device__ inline const Spinor<complex<num_type>>& operator()(int site) const {
        return d_psi[site];
    }

    __device__ inline Spinor<complex<num_type>>& at(int site, int color, int spin) {
        return d_psi[spinor_site_offset(site) + color * 4 + spin];
    }
};


struct CGResult {
    int iterations;
    float residual;
    bool converged;
};


template<typename T>
CGResult solve_cg_mdaggerm(FermionFieldView b, FermionFieldView x, LatticeView lat, int max_iter = 1000, T tolerance = T(1e-6));

                    


class GaugeField {

private:

    std::mt19937 engine;
    std::uniform_real_distribution<num_type> dist;
public:
    


    Matrix<complex<num_type>,3>* d_links;
    Matrix<complex<num_type>,3>* d_links_old;
    Matrix<complex<num_type>,3>* d_links_smeared;
    Matrix<complex<num_type>,3>* d_moms;
    curandState* d_rng_states;

    Spinor<complex<num_type>>* d_phi;
    Spinor<complex<num_type>>* d_chi;
    Spinor<complex<num_type>>* d_phi_backup;

    num_type* d_workspace;
    int* d_hopping;
    LatticeParams params;
    num_type dH=0;
    num_type kappa;
    num_type mu;


    GaugeField(int Lx, int Ly, int Lz, int Lt,num_type Beta, num_type kappa_=0.0, num_type mu_=0.0) {
        params.L[0] = Lx; params.L[1] = Ly; params.L[2] = Lz; params.L[3] = Lt;
        params.volume = Lx * Ly * Lz * Lt;
        params.beta=Beta;
        kappa = kappa_;
        mu = mu_;
        params.blocks= (params.volume+ params.threads - 1) / params.threads;
        unsigned long long seed = 1234ULL;
        unsigned long long seed_links = 4321ULL;
        unsigned long long seed_fermion = 5678ULL;
        size_t size = params.volume * 4 * sizeof(Matrix<complex<num_type>,3>);
        size_t size_hopping = params.volume*8 * sizeof(int);
        size_t size_rng= params.volume *sizeof(curandState);
        size_t size_density= params.volume * sizeof (num_type);
        size_t size_spinor = params.volume * sizeof(Spinor<complex<num_type>>);
        
        cudaMalloc(&d_links, size);
        cudaMalloc(&d_links_old, size);
        cudaMalloc(&d_links_smeared, size);
        cudaMalloc(&d_workspace,size_density);
        cudaMalloc(&d_moms, size);
        cudaMalloc(&d_hopping,size_hopping);
        cudaMalloc(&d_rng_states,size_rng);
        cudaMalloc(&d_phi, size_spinor);
        cudaMalloc(&d_chi, size_spinor);
        cudaMalloc(&d_phi_backup, size_spinor);
        
        init_hopping();
        init_links(0.5f,seed_links);
        init_rng(seed);
        init_pseudofermion(seed_fermion);
    }
    ~GaugeField() {
        cudaFree(d_links);
        cudaFree(d_links_old);
        cudaFree(d_links_smeared);
        cudaFree(d_moms);
        cudaFree(d_hopping);
        cudaFree(d_phi);
        cudaFree(d_chi);
        cudaFree(d_phi_backup);
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
        v.kappa = this->kappa;
        return v;
    }

    FermionFieldView fermion_view() {
        FermionFieldView fv;
        fv.d_psi = this->d_phi;
        fv.volume = this->params.volume;
        return fv;
    }

    FermionFieldView fermion_CGed_view() {
        FermionFieldView fv;
        fv.d_psi = this->d_chi;
        fv.volume = this->params.volume;
        return fv;
    }

    FermionFieldView fermion_backup_view() {
        FermionFieldView fv;
        fv.d_psi = this->d_phi_backup;
        fv.volume = this->params.volume;
        return fv;
    }

    void backup_fermion();
    void restore_fermion();
    void generate_pseudofermion();

    //initialize hopping and links
    void init_hopping();
    void init_links(num_type epsi, unsigned long seed);
    void init_rng(unsigned long long seed);
    void init_pseudofermion(unsigned long long seed);

    bool acc_rej(num_type H_i, num_type H_f);

    // Phys obsev 
    // num_type Hamilt(FermionFieldView chi);
    num_type Hamilt();
    num_type Clover_ene();
    num_type topo_charge();
    num_type calculate_plaquette();
    num_type calculate_wilson_loop( int r, int s,int mu, int nu);
    num_type fermion_action();
    num_type total_action();

    //HMC update
    void update_1step(num_type length, int num_steps);

    //Metropolis update
    num_type metropolis_update(num_type epsilon, int nsteps);

    //Smearing
    void smear_links(num_type alpha, int n_iter);
    void gradient_flow(num_type epsilon, int n_steps);

    //Fermion force
    void calculate_fermion_force();

    //Fermion field operations
    void apply_wilson_dagger_d(Spinor<complex<num_type>>* src, Spinor<complex<num_type>>* dst);
    void apply_wilson_d(Spinor<complex<num_type>>* src, Spinor<complex<num_type>>* dst);

    // Dynamic fermion HMC update
    void update_1step_dynferm(num_type length, int num_steps);
    num_type Hamilt_dynferm(num_type& out_gauge_action, num_type& out_fermion_action);

};

__global__ void kernel_calculate_plaquette(LatticeView lat, num_type* results);
__global__ void kernel_wilson_loop(LatticeView lat, num_type* results, int r, int s, int mu, int nu);
__global__ void kernel_smear_links(LatticeView lat, num_type alpha);
__global__ void kernel_gradient_flow(LatticeView lat, num_type epsilon);
__global__ void kernel_reunitarize_links(LatticeView lat);
__global__ void kernel_Energe(LatticeView lat, num_type* results);

// Fermion field management kernels
__global__ void kernel_init_pseudofermion(FermionFieldView phi, curandState* d_states);
__global__ void kernel_backup_fermion(FermionFieldView src, FermionFieldView dst);
__global__ void kernel_restore_fermion(FermionFieldView dst, FermionFieldView src);

// Wilson Dirac operator kernels (in dirac.cu)
__global__ void kernel_wilson_dslash(LatticeView lat, FermionFieldView src, FermionFieldView dst, int is_dagger);
__global__ void kernel_fermion_action(FermionFieldView phi, num_type* result);
__global__ void kernel_fermion_force(LatticeView lat, FermionFieldView phi, num_type factor);


}