#include "../include/gradient_flow.hpp"
#include "../include/observables.hpp"
#include "../include/gauge_ops.hpp"

namespace qcd {

// Traceless anti-Hermitian projection: TA(M) = (M - M†)/2 - Tr[(M - M†)/2]/3 * I
static SU3 TA(const SU3& M) {
    SU3 ah;
    for(int r=0;r<3;r++) for(int c=0;c<3;c++)
        ah(r,c) = (M(r,c) - std::conj(M(c,r))) * 0.5;
    complex tr = ah.trace() * (1.0/3.0);
    for(int i=0;i<3;i++) ah(i,i) -= tr;
    return ah;
}

// Compute flow force Z_mu(s) = TA(U_mu(s) * Sigma_mu(s))
// where Sigma is the staple sum. Stores in flat array Z[mu*vol + s].
static void compute_flow_force(GaugeField& gf, std::vector<SU3>& Z) {
    int vol = gf.layout.vol_local;
    gf.exchange_halo();
    Z.resize(NDIM * vol);
    #pragma omp parallel for schedule(static)
    for(int s=0; s<vol; s++) {
        for(int mu=0; mu<NDIM; mu++) {
            SU3 sigma = gf.staple(mu, s);
            Z[mu*vol+s] = TA(gf.link(mu,s) * sigma);
        }
    }
}

// Simple Euler integrator for Wilson flow (matching GPU implementation)
// Flow equation: dU/dt = -TA(U * Sigma) * U
// Euler step: U_new = U + (-eps * TA(U*Sigma)) * U
//
// GPU reference code does: Force = (M - M†) * (-eps/2)
//   which equals -eps * (M-M†)/2 = -eps * TA(M)  [same thing]
// So the coefficient for TA(M) is -eps (NOT -eps/2).
void flow_step(GaugeField& gf, double eps) {
    int vol = gf.layout.vol_local;
    std::vector<SU3> Z(NDIM * vol);

    compute_flow_force(gf, Z);

    #pragma omp parallel for schedule(static)
    for(int s=0; s<vol; s++) {
        for(int mu=0; mu<NDIM; mu++) {
            int idx = mu*vol + s;
            SU3 Force = Z[idx] * (-eps);   // -eps * TA(U*Sigma)
            gf.link(mu,s) = gf.link(mu,s) + Force * gf.link(mu,s);
            reunitarize(gf.link(mu,s));
        }
    }
}

// Run Wilson flow on a copy of gf_orig
void run_flow(const GaugeField& gf_orig, double eps, double t_max,
              void (*callback)(double t, double Q, double E, void* ctx), void* ctx)
{
    GaugeField gf = gf_orig;   // work on copy
    int n_steps = (int)(t_max / eps + 0.5);

    for(int i=0; i<n_steps; i++) {
        flow_step(gf, eps);
        double t = (i+1) * eps;
        gf.exchange_halo();
        double Q = topo_charge(gf);
        double E = clover_energy(gf);
        if(callback) callback(t, Q, E, ctx);
    }
}

} // namespace qcd
