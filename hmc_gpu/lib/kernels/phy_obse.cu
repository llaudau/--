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


__global__ void kernel_topo_charge(LatticeView lat, num_type* results) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lat.volume) return;

    num_type topo_sum = 0;
    constexpr num_type epsilon0123 = 1.0;
    constexpr num_type epsilon_0231 = -1.0;
    constexpr num_type epsilon_0312 = 1.0;
    constexpr num_type epsilon_0213 = -1.0;
    constexpr num_type epsilon_0312_check = 1.0;
    constexpr num_type epsilon_0132 = -1.0;
    constexpr num_type epsilon_0231_check = -1.0;
    constexpr num_type epsilon_0321 = 1.0;
    constexpr num_type epsilon_0123_check = 1.0;
    constexpr num_type epsilon_0213_check = -1.0;
    constexpr num_type epsilon_0312_final = 1.0;
    constexpr num_type epsilon_0132_final = -1.0;

    Matrix<complex<num_type>, 3> F[6];
    for (int i = 0; i < 6; i++) {
        F[i].setZero();
    }

    for (int nu = 1; nu < 4; nu++) {
        for (int mu = 0; mu < nu; mu++) {
            int clover_idx;
            if (mu == 0 && nu == 1) clover_idx = 0;
            else if (mu == 0 && nu == 2) clover_idx = 1;
            else if (mu == 0 && nu == 3) clover_idx = 2;
            else if (mu == 1 && nu == 2) clover_idx = 3;
            else if (mu == 1 && nu == 3) clover_idx = 4;
            else clover_idx = 5;

            int s_mu = lat.neighbor(site, mu);
            int s_nu = lat.neighbor(site, nu);
            int s_m_mu = lat.neighbor(site, mu + 4);
            int s_m_nu = lat.neighbor(site, nu + 4);

            Matrix<complex<num_type>, 3> U1 = lat.d_links[lat.link_idx(site, mu)];
            Matrix<complex<num_type>, 3> U2 = lat.d_links[lat.link_idx(s_mu, nu)];
            Matrix<complex<num_type>, 3> U3 = lat.d_links[lat.link_idx(s_nu, mu)].dagger();
            Matrix<complex<num_type>, 3> U4 = lat.d_links[lat.link_idx(s_m_nu, nu)].dagger();

            Matrix<complex<num_type>, 3> plaquette1 = U1 * U2 * U3 * U4;

            U1 = lat.d_links[lat.link_idx(s_mu, nu)];
            U2 = lat.d_links[lat.link_idx(s_mu, mu + 4)].dagger();
            U3 = lat.d_links[lat.link_idx(s_m_mu, nu)].dagger();
            U4 = lat.d_links[lat.link_idx(s_nu, mu)];

            Matrix<complex<num_type>, 3> plaquette2 = U1 * U2 * U3 * U4;

            U1 = lat.d_links[lat.link_idx(s_nu, mu)];
            U2 = lat.d_links[lat.link_idx(s_m_mu, nu)].dagger();
            U3 = lat.d_links[lat.link_idx(s_m_mu, mu + 4)];
            U4 = lat.d_links[lat.link_idx(s_m_nu, nu)];

            Matrix<complex<num_type>, 3> plaquette3 = U1 * U2 * U3 * U4;

            U1 = lat.d_links[lat.link_idx(s_m_nu, nu)].dagger();
            U2 = lat.d_links[lat.link_idx(s_m_nu, mu)];
            U3 = lat.d_links[lat.link_idx(s_nu, mu)];
            U4 = lat.d_links[lat.link_idx(site, nu)];

            Matrix<complex<num_type>, 3> plaquette4 = U1 * U2 * U3 * U4;

            F[clover_idx] = plaquette1 + plaquette2 + plaquette3 + plaquette4;
        }
    }

    topo_sum = epsilon0123 * (F[0] * F[3]).trace().real() +
               epsilon_0231 * (F[0] * F[4]).trace().real() +
               epsilon_0312 * (F[0] * F[5]).trace().real() +
               epsilon_0213 * (F[1] * F[3]).trace().real() +
               epsilon_0312_check * (F[1] * F[4]).trace().real() +
               epsilon_0132 * (F[1] * F[5]).trace().real() +
               epsilon_0231_check * (F[2] * F[3]).trace().real() +
               epsilon_0321 * (F[2] * F[4]).trace().real() +
               epsilon_0123_check * (F[2] * F[5]).trace().real() +
               epsilon_0213_check * (F[3] * F[5]).trace().real() +
               epsilon_0312_final * (F[4] * F[5]).trace().real();

    constexpr num_type norm_factor = 1.0 / (32.0 * M_PI * M_PI);
    results[site] = topo_sum * norm_factor;
}


num_type GaugeField::Hamilt(num_type &out_action){

    Ham_des<<<params.blocks,params.threads>>>(this->view(),d_workspace);
    thrust::device_ptr<num_type> ptr(d_workspace);

    out_action = thrust::reduce(thrust::device, ptr, ptr + params.volume, 0.0, thrust::plus<num_type>());

    num_type total_kinetic = thrust::reduce(thrust::device, ptr + params.volume, ptr + 2 * params.volume, 0.0, thrust::plus<num_type>());

    return out_action + total_kinetic;
}


num_type GaugeField::topo_charge() {
    kernel_topo_charge<<<params.blocks, params.threads>>>(this->view(), d_workspace);
    thrust::device_ptr<num_type> ptr(d_workspace);
    num_type topo = thrust::reduce(thrust::device, ptr, ptr + params.volume, 0.0, thrust::plus<num_type>());
    return topo;
}