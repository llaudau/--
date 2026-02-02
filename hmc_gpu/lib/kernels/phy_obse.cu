#include "lattice.cuh"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h> 



__global__ void Ham_des(LatticeView lat, num_type* results) {
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

    results[site+lat.volume] = kinetic; // H = S + K
}


num_type GaugeField::Hamilt(num_type &out_action){

    Ham_des<<<params.blocks,params.threads>>>(this->view(),d_workspace);
    thrust::device_ptr<num_type> ptr(d_workspace);

    out_action = thrust::reduce(thrust::device, ptr, ptr + params.volume, 0.0, thrust::plus<num_type>());

    // 3. Reduce the second half to get Total Kinetic Energy
    num_type total_kinetic = thrust::reduce(thrust::device, ptr + params.volume, ptr + 2 * params.volume, 0.0, thrust::plus<num_type>());

    // 4. Return total Hamiltonian
    return out_action + total_kinetic;
}