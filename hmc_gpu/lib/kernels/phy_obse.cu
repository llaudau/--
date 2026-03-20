#include "lattice.cuh"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h> 

namespace qcdcuda{


__global__ void kernel_calculate_plaquette(LatticeView lat, num_type* results) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lat.volume) return;

    num_type action = 0;

    // 1. Wilson Action (Plaquettes)
    for (int nu = 1; nu < 4; nu++) { // Starts from 1 to avoid double counting
        for (int mu = 0; mu < nu; mu++) {
            // Path: site -> site+mu -> site+mu+nu -> site+nu -> site
            int s_mu = lat.neighbor(site, mu);
            int s_nu = lat.neighbor(site, nu);

            Matrix<complex<num_type>, 3> U1 = lat.d_links[lat.link_idx(site, mu)];
            Matrix<complex<num_type>, 3> U2 = lat.d_links[lat.link_idx(s_mu, nu)];
            Matrix<complex<num_type>, 3> U3 = lat.d_links[lat.link_idx(s_nu, mu)];
            Matrix<complex<num_type>, 3> U4 = lat.d_links[lat.link_idx(site, nu)];
            Matrix<complex<num_type>, 3> Plaq = U1 * U2 * U3.dagger() * U4.dagger();

            action += (Plaq.trace()).real() / 3.0;
        }
    }
    results[site]=action;
}

__global__ void Ham_dens(LatticeView lat, num_type* results) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lat.volume) return;

    num_type action = 0;
    num_type kinetic = 0;

    // 1. Wilson Action (Plaquettes)
    for (int nu = 1; nu < 4; nu++) { // Starts from 1 to avoid double counting
        for (int mu = 0; mu < nu; mu++) {
            // Path: site -> site+mu -> site+mu+nu -> site+nu -> site
            int s_mu = lat.neighbor(site, mu);
            int s_nu = lat.neighbor(site, nu);

            Matrix<complex<num_type>, 3> U1 = lat.d_links[lat.link_idx(site, mu)];
            Matrix<complex<num_type>, 3> U2 = lat.d_links[lat.link_idx(s_mu, nu)];
            Matrix<complex<num_type>, 3> U3 = lat.d_links[lat.link_idx(s_nu, mu)];
            Matrix<complex<num_type>, 3> U4 = lat.d_links[lat.link_idx(site, nu)];
            Matrix<complex<num_type>, 3> Plaq = U1 * U2 * U3.dagger() * U4.dagger();

            action += (3.0 - (Plaq.trace()).real()) * lat.beta / 3.0;
        }
    }
    results[site]=action;
    // 2. Kinetic Energy (Momenta)
    for (int mu = 0; mu < 4; mu++) {
        Matrix<complex<num_type>, 3> P = lat.d_moms[lat.link_idx(site, mu)];
        kinetic += (P * P).trace().real();

    }

    results[site] += kinetic; // H = S + K
}


__global__ void kernel_topo_charge(LatticeView lat, num_type* results) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lat.volume) return;

    num_type topo_sum = 0;
    constexpr num_type epsilon_0123 = 1.0;
    constexpr num_type epsilon_0213 = -1.0;
    constexpr num_type epsilon_0312 = 1.0;
    // constexpr num_type epsilon_0132_final = -1.0;

    Matrix<complex<num_type>, 3> F[6];
    for (int i = 0; i < 6; i++) {
        F[i].setZero();
    }

    //clover definition
    for (int nu = 1; nu < 4; nu++) {
        for (int mu = 0; mu < nu; mu++) {
            int clover_idx;
            if (mu == 0 && nu == 1) clover_idx = 0;
            else if (mu == 0 && nu == 2) clover_idx = 1;
            else if (mu == 0 && nu == 3) clover_idx = 2;
            else if (mu == 1 && nu == 2) clover_idx = 3;
            else if (mu == 1 && nu == 3) clover_idx = 4;
            else clover_idx = 5;

            int sites[4]={site,lat.neighbor(site,mu+4),lat.neighbor(site,nu+4),lat.neighbor(lat.neighbor(site,mu+4),nu+4)};

            for (int i=0; i<4;i++){
                int s=sites[i];
                int s_mu = lat.neighbor(s, mu);
                int s_nu = lat.neighbor(s, nu);
                Matrix<complex<num_type>, 3> U1 = lat.d_links[lat.link_idx(s, mu)];
                Matrix<complex<num_type>, 3> U2 = lat.d_links[lat.link_idx(s_mu, nu)];
                Matrix<complex<num_type>, 3> U3 = lat.d_links[lat.link_idx(s_nu, mu)].dagger();
                Matrix<complex<num_type>, 3> U4 = lat.d_links[lat.link_idx(s, nu)].dagger();
                Matrix<complex<num_type>, 3> plaquette = U1 * U2 * U3 * U4;
                F[clover_idx] += (plaquette.imag()) *complex<num_type>(1.0/4.0,0);
                // F[clover_idx] += (plaquette-plaquette.dagger()) *complex<num_type>(1,-1.0/8.0);
                
            complex<num_type> trF=F[clover_idx].trace()*complex<num_type>(1.0/3.0);
            F[clover_idx](0,0)=F[clover_idx](0,0)-trF;
            F[clover_idx](1,1)=F[clover_idx](1,1)-trF;
            F[clover_idx](2,2)=F[clover_idx](2,2)-trF;
            
            }
        }
    }
    topo_sum += epsilon_0123 * (F[0] * F[5]).trace().real() ;
    topo_sum += epsilon_0213 * (F[1] * F[4]).trace().real() ;
    topo_sum += epsilon_0312 * (F[2] * F[3]).trace().real() ;

    constexpr num_type norm_factor = 1.0 / (16.0 * M_PI * M_PI);
    results[site] = topo_sum * norm_factor;
    }




__global__ void kernel_Energe(LatticeView lat, num_type* results) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lat.volume) return;

    num_type ener_sum = 0;
    // constexpr num_type epsilon_0132_final = -1.0;

    Matrix<complex<num_type>, 3> F[6];
    for (int i = 0; i < 6; i++) {
        F[i].setZero();
    }

    //clover definition
    for (int nu = 1; nu < 4; nu++) {
        for (int mu = 0; mu < nu; mu++) {
            int clover_idx;
            if (mu == 0 && nu == 1) clover_idx = 0;
            else if (mu == 0 && nu == 2) clover_idx = 1;
            else if (mu == 0 && nu == 3) clover_idx = 2;
            else if (mu == 1 && nu == 2) clover_idx = 3;
            else if (mu == 1 && nu == 3) clover_idx = 4;
            else clover_idx = 5;

            int sites[4]={site,lat.neighbor(site,mu+4),lat.neighbor(site,nu+4),lat.neighbor(lat.neighbor(site,mu+4),nu+4)};

            for (int i=0; i<4;i++){
                int s=sites[i];
                int s_mu = lat.neighbor(s, mu);
                int s_nu = lat.neighbor(s, nu);
                Matrix<complex<num_type>, 3> U1 = lat.d_links[lat.link_idx(s, mu)];
                Matrix<complex<num_type>, 3> U2 = lat.d_links[lat.link_idx(s_mu, nu)];
                Matrix<complex<num_type>, 3> U3 = lat.d_links[lat.link_idx(s_nu, mu)].dagger();
                Matrix<complex<num_type>, 3> U4 = lat.d_links[lat.link_idx(s, nu)].dagger();
                Matrix<complex<num_type>, 3> plaquette = U1 * U2 * U3 * U4;
                F[clover_idx] += (plaquette.imag()) *complex<num_type>(1.0/4.0,0);
                // F[clover_idx] += (plaquette-plaquette.dagger()) *complex<num_type>(0.0,-1.0/8.0);
                
            complex<num_type> trF=F[clover_idx].trace()*complex<num_type>(1.0/3.0);
            F[clover_idx](0,0)=F[clover_idx](0,0)-trF;
            F[clover_idx](1,1)=F[clover_idx](1,1)-trF;
            F[clover_idx](2,2)=F[clover_idx](2,2)-trF;
            
            }
        }
    }
    ener_sum +=   (F[0] * F[0]).trace().real() ;
    ener_sum +=   (F[1] * F[1]).trace().real() ;
    ener_sum +=   (F[2] * F[2]).trace().real() ;
    ener_sum +=   (F[3] * F[3]).trace().real() ;
    ener_sum +=   (F[4] * F[4]).trace().real() ;
    ener_sum +=   (F[5] * F[5]).trace().real() ;

    constexpr num_type norm_factor = 1.0 / (2.0);
    results[site] = ener_sum * norm_factor *2;
    }

num_type GaugeField::Clover_ene(){
    num_type Ham=0.0;
    kernel_Energe<<<params.blocks,params.threads>>>(this->view(),d_workspace);
    thrust::device_ptr<num_type> ptr(d_workspace);
    Ham = thrust::reduce(thrust::device, ptr, ptr + params.volume, 0.0, thrust::plus<num_type>());
    return Ham/params.volume;
}


__global__ void kernel_fermion_action(FermionFieldView phi, FermionFieldView chi, num_type* action_out) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;

    num_type action = 0.0;

    if (site < phi.volume) {
        Spinor<complex<num_type>>& phi_site = phi(site);
        Spinor<complex<num_type>>& chi_site = chi(site);

        complex<num_type> s_f = complex<num_type>(0, 0);
        for(int c = 0; c < 3; c++) {
            for(int s = 0; s < 4; s++) {
                s_f += conj(phi_site(c, s)) * chi_site(c, s);
            }
        }
        action = s_f.real();
    }

    action_out[site] = action;
}

// num_type GaugeField::Hamilt(FermionFieldView chi){
num_type GaugeField::Hamilt(){
    num_type Ham=0.0;
    Ham_dens<<<params.blocks,params.threads>>>(this->view(),d_workspace);
    thrust::device_ptr<num_type> ptr(d_workspace);
    Ham = thrust::reduce(thrust::device, ptr, ptr + params.volume, 0.0, thrust::plus<num_type>());

    // kernel_fermion_action<<<params.blocks,params.threads>>>(fermion_view(),chi,d_workspace);
    return Ham;
}


num_type GaugeField::topo_charge() {
    kernel_topo_charge<<<params.blocks, params.threads>>>(this->view(), d_workspace);
    thrust::device_ptr<num_type> ptr(d_workspace);
    num_type topo = thrust::reduce(thrust::device, ptr, ptr + params.volume, 0.0, thrust::plus<num_type>());
    return topo;
}


num_type GaugeField::calculate_plaquette() {
    kernel_calculate_plaquette<<<params.blocks, params.threads>>>(this->view(), d_workspace);
    thrust::device_ptr<num_type> ptr(d_workspace);
    num_type avg_plaq = thrust::reduce(thrust::device, ptr, ptr + params.volume, (num_type)0.0, thrust::plus<num_type>());
    return avg_plaq / params.volume;
}


__global__ void kernel_wilson_loop(LatticeView lat, num_type* results, int r, int s, int mu, int nu) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lat.volume) return;

    num_type wloop = 0.0;

    Matrix<complex<num_type>, 3> prod;
    prod.setIdentity();

    int current_site = site;
    // Go forward r steps in mu direction
    for (int i = 0; i < r; i++) {
        prod = prod * lat.d_links[lat.link_idx(current_site, mu)];
        current_site = lat.neighbor(current_site, mu);
    }

    // Go forward s steps in nu direction
    for (int i = 0; i < s; i++) {
        prod = prod * lat.d_links[lat.link_idx(current_site, nu)];
        current_site = lat.neighbor(current_site, nu);
    }

    // Go backward r steps in mu direction (using mu+4 for backward)
    for (int i = 0; i < r; i++) {
        current_site = lat.neighbor(current_site, mu + 4);
        prod = prod * lat.d_links[lat.link_idx(current_site, mu)].dagger();
    }

    // Go backward s steps in nu direction (using nu+4 for backward)
    for (int i = 0; i < s; i++) {
        current_site = lat.neighbor(current_site, nu + 4);
        prod = prod * lat.d_links[lat.link_idx(current_site, nu)].dagger();
    }

    wloop = (prod.trace()).real() / 3.0;
    results[site] = wloop;
}



num_type GaugeField::calculate_wilson_loop( int r, int s,int mu, int nu) {
    kernel_wilson_loop<<<params.blocks, params.threads>>>(this->view(), d_workspace, r, s,mu, nu);
    thrust::device_ptr<num_type> ptr(d_workspace);
    num_type wilson_sum = thrust::reduce(thrust::device, ptr, ptr + params.volume, (num_type)0.0, thrust::plus<num_type>());
    
    int num_loops = params.volume ;

    return wilson_sum / num_loops;

}



}