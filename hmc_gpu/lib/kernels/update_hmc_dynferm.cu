#include "lattice.cuh"
#include "dirac.cuh"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

namespace qcdcuda {

using namespace su3_generators;

__global__ void update_force(LatticeView lat, num_type epsi);
__global__ void kernel_update_gauge(LatticeView lat, num_type epsilon);




__global__ void kernel_refresh_mom(LatticeView lat, curandState* d_states){
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site>=lat.volume) return;
    curandState localState = d_states[site];
    for (int mu=0;mu<4; mu++){
        int link_idx = site + mu * lat.volume;
        generate_gaussian_su3_algebra(lat.d_moms[link_idx],&localState);
    }
    d_states[site]=localState;
}


__global__ void update_mom(LatticeView lat,num_type epsi){
    int site = blockIdx.x*blockDim.x+threadIdx.x; 
    if (site>=lat.volume) return;
    for( int mu=0;mu<4;mu++){
        int link_index=lat.link_idx(site,mu);

        lat.d_links[link_index]+=complex<num_type>(0.0,epsi)*lat.d_moms[link_index]*lat.d_links[link_index]+complex<num_type>(-epsi*epsi/2,0)*lat.d_moms[link_index]*lat.d_moms[link_index]*lat.d_links[link_index];
        // check reunitarize  
        reunitarize(lat.d_links[link_index]);

    }
}


__global__ void update_force(LatticeView lat, num_type epsi) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lat.volume) return;

    complex<num_type> coeff ;
    coeff=complex<num_type>(0.0,-epsi * lat.beta / 12.0);
    for (int mu = 0; mu < 4; mu++) {
        int idx_mu = lat.link_idx(site, mu);
        auto U_mu = lat.d_links[idx_mu]; // LOAD ONCE to register
        
        Matrix<complex<num_type>, 3> Staple;
        Staple.setZero();

        for (int nu = 0; nu < 4; nu++) {
            if (nu == mu) continue;
            
            // Pre-calculate neighbor indices to avoid repeated d_hopping reads
            int site_p_mu = lat.neighbor(site, mu);
            int site_p_nu = lat.neighbor(site, nu);
            int site_m_nu = lat.neighbor(site, nu + 4);
            int site_p_mu_m_nu = lat.neighbor(site_p_mu, nu + 4);

            // Upper Staple
            Staple += lat.d_links[lat.link_idx(site_p_mu, nu)] * lat.d_links[lat.link_idx(site_p_nu, mu)].dagger() * lat.d_links[lat.link_idx(site, nu)].dagger();

            // Lower Staple
            Staple += lat.d_links[lat.link_idx(site_p_mu_m_nu, nu)].dagger() * lat.d_links[lat.link_idx(site_m_nu, mu)].dagger() * lat.d_links[lat.link_idx(site_m_nu, nu)];
        }
        

        // Force = TA(U * Staple_dag)
        Matrix<complex<num_type>, 3> Temp = U_mu * Staple;
        
        // This is the Traceless Anti-Hermitian projection
        Matrix< complex <num_type>, 3> Force = (Temp - Temp.dagger())* coeff;
        complex<num_type> tr_third = Force.trace() *(num_type) (1.0/3.0);

        // 2. Subtract only from the diagonal
        Force(0,0) =Force(0,0)- tr_third;
        Force(1,1) =Force(1,1)- tr_third;
        Force(2,2) =Force(2,2)- tr_third;
        lat.d_moms[idx_mu] =lat.d_moms[idx_mu]- Force ;
        
        
        
    }

}


// CG solver forward declarations
// CGResult solve_cg_mdaggerm(FermionFieldView b, FermionFieldView x, LatticeView lat, int max_iter, float tolerance);

__global__ void kernel_generate_gaussian_eta(FermionFieldView eta, curandState* d_states) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= eta.volume) return;

    Spinor<complex<num_type>>& eta_site = eta(site);
    curandState state = d_states[site];

    for(int c = 0; c < 3; c++) {
        for(int s = 0; s < 4; s++) {
            float x = curand_uniform(&state);
            float y = curand_uniform(&state);
            eta_site(c, s) = complex<num_type>(
                sqrtf(-2.0f * logf(x)) * cosf(2.0f * M_PI * y),
                sqrtf(-2.0f * logf(x)) * sinf(2.0f * M_PI * y)
            );
        }
    }

    d_states[site] = state;
}

__global__ void kernel_transform_eta_to_phi(LatticeView lat, FermionFieldView eta, FermionFieldView phi) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lat.volume) return;

    Spinor<complex<num_type>> chi;
    chi.setZero();

    for (int mu = 0; mu < 4; mu++) {
        int site_fwd = lat.neighbor(site, mu);
        Spinor<complex<num_type>>& eta_fwd = eta(site_fwd);
        Matrix<complex<num_type>, 3> U_fwd = lat.d_links[lat.link_idx(site, mu)];
        wilson_hop(chi, U_fwd, eta_fwd, mu, lat.kappa, +1);

        int site_bwd = lat.neighbor(site, mu + 4);
        Spinor<complex<num_type>>& eta_bwd = eta(site_bwd);
        Matrix<complex<num_type>, 3> U_bwd = lat.d_links[lat.link_idx(site_bwd, mu)].dagger();
        wilson_hop(chi, U_bwd, eta_bwd, mu, lat.kappa, -1);
    }

    Spinor<complex<num_type>>& phi_site = phi(site);
    // apply_gamma5(phi_site, chi);
}

__global__ void kernel_refresh_fermion_mom(LatticeView lat, curandState* d_states) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lat.volume) return;

    curandState state = d_states[site];

    for(int i = 0; i < 9; i++) {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        (void)x;
        (void)y;
    }

    d_states[site] = state;
}

__global__ void kernel_fermion_force_dyn(LatticeView lat, FermionFieldView phi, FermionFieldView chi, num_type factor) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lat.volume) return;

    num_type kappa = lat.kappa;

    for(int mu = 0; mu < 4; mu++) {
        Matrix<complex<num_type>, 3> force_mat;
        force_mat.setZero();

        Spinor<complex<num_type>> psi_fwd;
        psi_fwd.setZero();
        int site_fwd = lat.neighbor(site, mu);
        psi_fwd.copy(chi(site_fwd));

        Matrix<complex<num_type>, 3> U_mu = lat.d_links[lat.link_idx(site, mu)];
        Matrix<complex<num_type>, 3> U_mu_dag = U_mu.dagger();

        for(int a = 0; a < 8; a++) {
            Matrix<complex<num_type>, 3> Ta = generator<num_type>(a);

            Spinor<complex<num_type>> temp1;
            temp1.setZero();
            wilson_hop(temp1, U_mu, psi_fwd, mu, kappa, +1);

            Spinor<complex<num_type>> chi_local = chi(site);
            complex<num_type> coeff1 = complex<num_type>(0, 0);
            for(int c = 0; c < 3; c++) {
                for(int s = 0; s < 4; s++) {
                    coeff1 += conj(chi_local(c, s)) * temp1(c, s);
                }
            }
            coeff1 *= complex<num_type>(0, -2 * kappa);

            Spinor<complex<num_type>> psi_bwd;
            psi_bwd.setZero();
            int site_bwd = lat.neighbor(site, mu + 4);
            psi_bwd.copy(chi(site_bwd));

            Spinor<complex<num_type>> temp2;
            temp2.setZero();
            wilson_hop(temp2, U_mu_dag, psi_bwd, mu, kappa, -1);

            complex<num_type> coeff2 = complex<num_type>(0, 0);
            for(int c = 0; c < 3; c++) {
                for(int s = 0; s < 4; s++) {
                    coeff2 += conj(chi_local(c, s)) * temp2(c, s);
                }
            }
            coeff2 *= complex<num_type>(0, -2 * kappa);

            Matrix<complex<num_type>, 3> force_contrib;
            force_contrib.setZero();
            for(int i = 0; i < 3; i++) {
                for(int j = 0; j < 3; j++) {
                    force_contrib(i, j) = (coeff1 + coeff2) * Ta(i, j);
                }
            }
            force_mat += force_contrib;
        }

        int link_idx = lat.link_idx(site, mu);
        Matrix<complex<num_type>, 3>& P = lat.d_moms[link_idx];

        Matrix<complex<num_type>, 3> P_new = P - complex<num_type>(factor, 0.0) * force_mat;

        complex<num_type> tr = P_new.trace();
        complex<num_type> tr_part = tr / complex<num_type>(3.0, 0.0);

        for(int i = 0; i < 3; i++) {
            P_new(i, i) -= tr_part;
        }

        P = P_new;
    }
}



void GaugeField::update_1step_dynferm(num_type length, int num_steps) {
    num_type eps = length / num_steps;
    num_type eps_half = eps / 2.0;
    num_type eps_full = eps;

    cudaMemcpy(d_links_old, d_links,
               params.volume * 4 * sizeof(Matrix<complex<num_type>, 3>),
               cudaMemcpyDeviceToDevice);

    kernel_generate_gaussian_eta<<<params.blocks, params.threads>>>(
        fermion_view(), d_rng_states);

    kernel_transform_eta_to_phi<<<params.blocks, params.threads>>>(
        view(), fermion_view(), fermion_view());

    kernel_refresh_mom<<<params.blocks, params.threads>>>(view(), d_rng_states);

    num_type* d_kinetic;
    num_type* d_action_gauge;
    num_type* d_action_fermion;

    cudaMalloc(&d_kinetic, params.volume * sizeof(num_type));
    cudaMalloc(&d_action_gauge, params.volume * sizeof(num_type));
    cudaMalloc(&d_action_fermion, params.volume * sizeof(num_type));


    


    //calculate the H before step

    solve_cg_mdaggerm(fermion_view(), fermion_CGed_view(), view(), 1000, (num_type)1e-6);
    num_type Hi=0;
    Hi=Hamilt(fermion_CGed_view());



    thrust::device_ptr<num_type> ptr_action_fermion(d_action_fermion);



    //update 
    update_force<<<params.blocks, params.threads>>>(view(), eps_half);
    kernel_fermion_force_dyn<<<params.blocks, params.threads>>>(view(), fermion_view(), fermion_CGed_view(), eps_half);
    for(int i = 0; i < num_steps; i++) {

        update_mom<<<params.blocks, params.threads>>>(view(), eps_full);

        //solve CGed spinor and calculate force
        solve_cg_mdaggerm(fermion_view(), fermion_CGed_view(), view(), 1000, (num_type)1e-6);

        update_force<<<params.blocks, params.threads>>>(view(), eps_full);
        kernel_fermion_force_dyn<<<params.blocks, params.threads>>>(view(), fermion_view(), fermion_CGed_view(), eps_full);

    }

    update_mom<<<params.blocks, params.threads>>>(view(), eps_full);
    solve_cg_mdaggerm(fermion_view(), fermion_CGed_view(), view(), 1000, (num_type)1e-6);

    update_force<<<params.blocks, params.threads>>>(view(), eps_half);
    kernel_fermion_force_dyn<<<params.blocks, params.threads>>>(view(), fermion_view(), fermion_CGed_view(), eps_half);



    //calculate final Hf
    kernel_fermion_force_dyn<<<params.blocks, params.threads>>>(view(), fermion_view(), fermion_CGed_view(), eps_half);


    
    num_type Hf=0;
    Hf=Hamilt(fermion_CGed_view());
    if(!acc_rej(Hi, Hf)) {
        cudaMemcpy(d_links, d_links_old,
                   params.volume * 4 * sizeof(Matrix<complex<num_type>, 3>),
                   cudaMemcpyDeviceToDevice);
    }

}


} // namespace qcdcuda
