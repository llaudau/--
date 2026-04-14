#include "../include/hmc.hpp"
#include "../include/gauge_ops.hpp"
#include <cmath>
#include <random>
#include <iostream>

namespace qcd {

void refresh_momenta(GaugeField& gf, std::mt19937_64& rng) {
    int vol = gf.layout.vol_local;
    for(int mu=0; mu<NDIM; mu++)
        for(int s=0; s<vol; s++)
            gf.mom(mu, s) = gaussian_su3_algebra(rng);
}

// Traceless anti-Hermitian projection: TA(M) = (M - M^dag)/2 - Tr(M-M^dag)/6 * I
static SU3 TA(const SU3& M) {
    SU3 ah;
    for(int r=0;r<3;r++) for(int c=0;c<3;c++)
        ah(r,c) = (M(r,c) - std::conj(M(c,r))) * 0.5;
    complex tr = ah.trace() * (1.0/3.0);
    for(int i=0;i<3;i++) ah(i,i) -= tr;
    return ah;
}

// Force update: P += i*(eps*beta/6)*TA(U*Sigma)
// Convention A: P is Hermitian traceless. TA() returns anti-Hermitian.
// Multiplying anti-Hermitian by i gives Hermitian, preserving P's structure.
void update_force(GaugeField& gf, double eps) {
    int vol = gf.layout.vol_local;
    double coeff = eps * gf.params.beta / 6.0;

    gf.exchange_halo();

    #pragma omp parallel for schedule(static)
    for(int s=0; s<vol; s++) {
        for(int mu=0; mu<NDIM; mu++) {
            SU3 sigma = gf.staple(mu, s);
            SU3 F = TA(gf.link(mu, s) * sigma);       // anti-Hermitian
            gf.mom(mu, s) += F * complex(0.0, coeff);  // P += i*(ε*β/6)*TA(U*Σ)
        }
    }
}

// U_new = exp(i*eps*P) * U
// Convention A: P is Hermitian, so i*eps*P is anti-Hermitian → exp(...) ∈ SU(3)
// Approximated as I + i*eps*P + (i*eps*P)^2/2, then reunitarize
void update_links(GaugeField& gf, double eps) {
    int vol = gf.layout.vol_local;
    #pragma omp parallel for schedule(static)
    for(int s=0; s<vol; s++) {
        for(int mu=0; mu<NDIM; mu++) {
            SU3& U = gf.link(mu, s);
            const SU3& P = gf.mom(mu, s);
            SU3 iepsP = P * complex(0.0, eps);   // i*eps*P, anti-Hermitian ✓
            SU3 expP = SU3::identity() + iepsP + (iepsP * iepsP) * 0.5;
            U = expP * U;
            reunitarize(U);
        }
    }
}

// Leapfrog integrator:
// force(eps/2) -> [link(eps) -> force(eps)] x (N-1) -> link(eps) -> force(eps/2)
void leapfrog(GaugeField& gf, double traj_len, int md_steps) {
    double eps = traj_len / md_steps;
    update_force(gf, eps * 0.5);
    for(int i=0; i<md_steps-1; i++) {
        update_links(gf, eps);
        update_force(gf, eps);
    }
    update_links(gf, eps);
    update_force(gf, eps * 0.5);
}

// Local Hamiltonian contribution (each rank sums its own sites)
// H = beta/3 * sum_plaq (1 - Re Tr U_plaq/3) + 0.5 * sum Tr[P^2]
// We return the full global H via MPI_Allreduce
double hamiltonian(const GaugeField& gf) {
    int vol = gf.layout.vol_local;
    double beta = gf.params.beta;
    double H_local = 0.0;

    // Gauge action: S = beta * sum (1 - Re Tr[U_plaq]/3)  (standard Wilson)
    // Kinetic: K = Tr[P^2]  (in algebra coordinates: (1/2)*sum pi_a^2)
    //   Convention: P is Hermitian traceless, Tr[P^2] >= 0
    //   Link update uses dU/dt = i*P*U (follows from dK_alg/d(pi_a) = pi_a)
    for(int s=0; s<vol; s++) {
        // Plaquettes
        for(int mu=0; mu<NDIM; mu++) {
            for(int nu=mu+1; nu<NDIM; nu++) {
                int s_mu = gf.neighbor(s, mu);
                int s_nu = gf.neighbor(s, nu);
                const SU3& Umu  = gf.link_at(mu, s);
                const SU3& Unu_smu = gf.link_at(nu, s_mu);
                const SU3& Umu_snu = gf.link_at(mu, s_nu);
                const SU3& Unu  = gf.link_at(nu, s);
                SU3 plaq = Umu * Unu_smu * Umu_snu.dagger() * Unu.dagger();
                H_local += beta * (1.0 - plaq.re_trace() / 3.0);
            }
        }
        // Kinetic
        for(int mu=0; mu<NDIM; mu++) {
            const SU3& P = gf.mom(mu, s);
            SU3 P2 = P * P;
            H_local += P2.re_trace();  // K = Tr[P²]
        }
    }

    double H_global = 0.0;
    MPI_Allreduce(&H_local, &H_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return H_global;
}

// Full HMC step: returns 1 if accepted
int hmc_step(GaugeField& gf, std::mt19937_64& rng,
             double traj_len, int md_steps)
{
    // Save old config
    GaugeField gf_old = gf;

    // Refresh momenta
    refresh_momenta(gf, rng);

    double H_old = hamiltonian(gf);

    // Leapfrog
    leapfrog(gf, traj_len, md_steps);

    gf.exchange_halo();  // ensure ghosts are fresh after leapfrog
    double H_new = hamiltonian(gf);
    double dH = H_new - H_old;

    // Debug: print dH for first few steps (only rank 0)
    if(gf.layout.rank == 0) {
        static int dbg_count = 0;
        if(dbg_count++ < 5) {
            printf("  [HMC debug] H_old=%.6f H_new=%.6f dH=%.6f\n", H_old, H_new, dH);
            fflush(stdout);
        }
    }

    // Metropolis accept/reject (only rank 0 draws the random number, then broadcast)
    int accept = 0;
    if(gf.layout.rank == 0) {
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        double r = uni(rng);
        accept = (r < std::exp(-dH)) ? 1 : 0;
    }
    MPI_Bcast(&accept, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(!accept)
        gf = gf_old;

    return accept;
}

} // namespace qcd
