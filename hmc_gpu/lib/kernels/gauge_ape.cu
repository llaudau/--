#include "lattice.cuh"

namespace qcdcuda{





__global__ void kernel_smear_links(LatticeView lat, num_type alpha) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lat.volume) return;

    num_type coeff = alpha / 6.0;
    num_type selfcoeff = 1.0-alpha;
    
    for (int mu = 0; mu < 4; mu++) {
        Matrix<complex<num_type>, 3> staple;
        staple.setZero();

        for (int nu = 0; nu < 4; nu++) {
            if (nu == mu) continue;

            int site_p_mu = lat.neighbor(site, mu);
            int site_p_nu = lat.neighbor(site, nu);
            int site_m_nu = lat.neighbor(site, nu+4);
            int site_p_mu_m_nu=lat.neighbor(site_p_mu,nu+4);
            Matrix<complex<num_type>, 3> staple1 = lat.d_links[lat.link_idx(site, nu)] *
             lat.d_links[lat.link_idx(site_p_nu, mu)] * lat.d_links[lat.link_idx(site_p_mu, nu)].dagger();

            
            Matrix<complex<num_type>, 3> staple2 = lat.d_links[lat.link_idx(site_m_nu, nu)].dagger() *
             lat.d_links[lat.link_idx(site_m_nu, mu)] * lat.d_links[lat.link_idx(site_p_mu_m_nu, nu)];

            staple += staple1;
            staple += staple2;
        }

        Matrix<complex<num_type>, 3> coeff_staple = staple * complex<num_type>(coeff);
        Matrix<complex<num_type>, 3> smeared = lat.d_links[lat.link_idx(site, mu)]*complex<num_type>(selfcoeff) + coeff_staple;
        lat.d_links[lat.link_idx(site, mu)] = smeared;
    }
}




void GaugeField::smear_links(num_type alpha, int n_iter) {
    for (int iter = 0; iter < n_iter; iter++) {
        kernel_smear_links<<<params.blocks, params.threads>>>(this->view(), alpha);
        // Reunitarize after each smearing step to stay in SU(3)
        kernel_reunitarize_links<<<params.blocks, params.threads>>>(this->view());
    }
}

















}