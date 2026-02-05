#include "lattice.cuh"
#include "gauge_operation.cuh"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#define KERNEL_CHECK(name) { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("Error launching %s: %s\n", name, cudaGetErrorString(err)); \
    } \
}

__global__ void kernel_calculate_plaquette(LatticeView lat, num_type* results) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lat.volume) return;

    num_type plaq_sum = 0;
    int count = 0;

    for (int nu = 1; nu < 4; nu++) {
        for (int mu = 0; mu < nu; mu++) {
            int s_mu = lat.neighbor(site, mu);
            int s_nu = lat.neighbor(site, nu);

            Matrix<complex<num_type>, 3> U1 = lat.d_links[lat.link_idx(site, mu)];
            Matrix<complex<num_type>, 3> U2 = lat.d_links[lat.link_idx(s_mu, nu)];
            Matrix<complex<num_type>, 3> U3 = lat.d_links[lat.link_idx(s_nu, mu)];
            Matrix<complex<num_type>, 3> U4 = lat.d_links[lat.link_idx(site, nu)];

            Matrix<complex<num_type>, 3> Plaq = U1 * U2 * U3.dagger() * U4.dagger();
            plaq_sum += (Plaq.trace()).real();
            count++;
        }
    }

    results[site] = plaq_sum / (3.0 * count);
}


__global__ void kernel_metropolis(LatticeView lat, num_type epsilon, curandState* d_states, int* d_accept) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lat.volume) return;

    curandState localState = d_states[site];

    for (int mu = 0; mu < 4; mu++) {
        int link_idx = lat.link_idx(site, mu);
        Matrix<complex<num_type>, 3> U_old = lat.d_links[link_idx];

        Matrix<complex<num_type>, 3> V;
        generate_random_su3_near_identity(V, epsilon, &localState);

        Matrix<complex<num_type>, 3> U_new = U_old * V;

        Matrix<complex<num_type>, 3> Staple;
        Staple.setZero();

        for (int nu = 0; nu < 4; nu++) {
            if (nu == mu) continue;

            int site_p_mu = lat.neighbor(site, mu);
            int site_p_nu = lat.neighbor(site, nu);
            int site_m_nu = lat.neighbor(site, nu + 4);
            int site_p_mu_m_nu = lat.neighbor(site_p_mu, nu + 4);

            Staple += lat.d_links[lat.link_idx(site_p_mu, nu)] *
                      lat.d_links[lat.link_idx(site_p_nu, mu)].dagger() *
                      lat.d_links[lat.link_idx(site, nu)].dagger();

            Staple += lat.d_links[lat.link_idx(site_p_mu_m_nu, nu)].dagger() *
                      lat.d_links[lat.link_idx(site_m_nu, mu)].dagger() *
                      lat.d_links[lat.link_idx(site_m_nu, nu)];
        }

        num_type S_old = 0;
        num_type S_new = 0;

        Matrix<complex<num_type>, 3> UStaple_old = U_old * Staple;
        Matrix<complex<num_type>, 3> UStaple_new = U_new * Staple;

        S_old = - (lat.beta / 3.0) * (UStaple_old.trace()).real();
        S_new = - (lat.beta / 3.0) * (UStaple_new.trace()).real();

        num_type dS = S_new - S_old;

        int accept = 0;
        if (dS <= 0) {
            accept = 1;
        } else {
            num_type rand_val = curand_uniform(&localState);
            if (exp(-dS) > rand_val) {
                accept = 1;
            }
        }

        if (accept) {
            lat.d_links[link_idx] = U_new;
            reunitarize(lat.d_links[link_idx]);
            atomicAdd(d_accept, 1);
        }
    }

    d_states[site] = localState;
}


num_type GaugeField::metropolis_update(num_type epsilon, int nsteps) {
    int* d_accept;
    int accept_count = 0;
    cudaMalloc(&d_accept, sizeof(int));
    cudaMemset(d_accept, 0, sizeof(int));

    for (int step = 0; step < nsteps; step++) {
        kernel_metropolis<<<params.blocks, params.threads>>>(this->view(), epsilon, d_rng_states, d_accept);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(&accept_count, d_accept, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_accept);

    num_type total_links = params.volume * 4 * nsteps;
    return static_cast<num_type>(accept_count) / total_links;
}
