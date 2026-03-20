#include "lattice.cuh"

namespace qcdcuda{











__global__ void  kernel_gradient_flow(LatticeView lat, num_type epsi)  {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lat.volume) return;

    complex<num_type> coeff ;
    coeff=complex<num_type>(-epsi * 1.0 / 2.0,0.0);
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
        lat.d_links[idx_mu] =lat.d_links[idx_mu]+Force*lat.d_links[idx_mu] ;
        reunitarize(lat.d_links[idx_mu]);
        
        
    }
}





void GaugeField::gradient_flow(num_type epsilon, int n_steps) {
    for (int step = 0; step < n_steps; step++) {
        kernel_gradient_flow<<<params.blocks, params.threads>>>(this->view(), epsilon);
    }
}

























}