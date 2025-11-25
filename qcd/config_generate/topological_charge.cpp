#include "lattice.h"
#include <cmath>
#include <Eigen/Dense>

// Helper: Compute plaquette U_{mu,nu}(x) = U_mu(x) U_nu(x+mu) U_mu^dag(x+nu) U_nu^dag(x)
SU3Matrix Lattice::compute_plaquette(Vector4i site, int mu, int nu) const {
    // Move in +mu direction from site
    Vector4i site_pmu = site;
    site_pmu[mu] = (site_pmu[mu] + 1) % Nt;  // Only time dir wraps at Nt
    if (mu > 0) site_pmu[mu] = (site_pmu[mu]) % Ns;
    
    // Move in +nu direction from site
    Vector4i site_pnu = site;
    site_pnu[nu] = (site_pnu[nu] + 1) % Nt;  // Only time dir wraps at Nt
    if (nu > 0) site_pnu[nu] = (site_pnu[nu]) % Ns;
    
    // Move in +nu then +mu direction from site
    Vector4i site_pmu_pnu = site;
    site_pmu_pnu[mu] = (site_pmu_pnu[mu] + 1) % Nt;
    if (mu > 0) site_pmu_pnu[mu] = (site_pmu_pnu[mu]) % Ns;
    site_pmu_pnu[nu] = (site_pmu_pnu[nu] + 1) % Nt;
    if (nu > 0) site_pmu_pnu[nu] = (site_pmu_pnu[nu]) % Ns;
    
    // U_mu(x) U_nu(x+mu) U_mu^dag(x+nu) U_nu^dag(x)
    SU3Matrix plaq = get_link(site, mu) * 
                     get_link(site_pmu, nu) * 
                     get_link(site_pnu, mu).adjoint() * 
                     get_link(site, nu).adjoint();
    
    return plaq;
}

// Helper: Extract field strength tensor from plaquette
// F_{mu,nu} = (1/(2ig)) [U_{mu,nu} - U_{mu,nu}^dag]
// For dimensionless lattice (g=1), we compute the anti-hermitian part
SU3Matrix Lattice::field_strength_from_plaquette(SU3Matrix plaq) const {
    // F = (1/2i) [U - U^dag] (anti-hermitian)
    SU3Matrix F = (plaq - plaq.adjoint()) / ComplexD(0.0, 2.0);
    return F;
}

// Compute topological charge density at a site using clover definition
// q(x) = (1/(32 pi^2)) Re[Tr(F_{mu,nu} F^*_{mu,nu})] where F* is dual
double Lattice::topological_charge_density(Vector4i site) const {
    // Compute all 6 independent plaquettes: (01), (02), (03), (12), (13), (23)
    SU3Matrix plaq[6];
    int idx = 0;
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = mu + 1; nu < 4; ++nu) {
            plaq[idx++] = compute_plaquette(site, mu, nu);
        }
    }
    
    // Extract field strength tensors
    SU3Matrix F[6];
    for (int i = 0; i < 6; ++i) {
        F[i] = field_strength_from_plaquette(plaq[i]);
    }
    
    // Compute dual field strength: *F_{mu,nu} = (1/2) eps_{mu,nu,rho,sigma} F_{rho,sigma}
    // For 4D: *F_{01} = F_{23}, *F_{02} = -F_{13}, *F_{03} = F_{12}
    //         *F_{23} = F_{01}, *F_{13} = -F_{02}, *F_{12} = F_{03}
    SU3Matrix F_dual[6];
    F_dual[0] = F[5];   // *F_{01} = F_{23}
    F_dual[1] = -F[4];  // *F_{02} = -F_{13}
    F_dual[2] = F[3];   // *F_{03} = F_{12}
    F_dual[3] = F[2];   // *F_{12} = F_{03}
    F_dual[4] = -F[1];  // *F_{13} = -F_{02}
    F_dual[5] = F[0];   // *F_{23} = F_{01}
    
    // Compute trace of F * F_dual
    // q(x) = (1/(32 pi^2)) Im[Tr(F_{mu,nu} *F_{mu,nu})]
    ComplexD trace_sum = ComplexD(0.0, 0.0);
    for (int i = 0; i < 6; ++i) {
        SU3Matrix product = F[i] * F_dual[i];
        trace_sum += product.trace();
    }
    
    // Topological charge density (1/(32 pi^2) factor)
    double q_density = trace_sum.imag() / (32.0 * M_PI * M_PI);
    
    return q_density;
}

// Compute total topological charge for the entire lattice
double Lattice::topological_charge() const {
    double Q_total = 0.0;
    
    Vector4i site;
    for (site[0] = 0; site[0] < Nt; ++site[0]) {
        for (site[1] = 0; site[1] < Ns; ++site[1]) {
            for (site[2] = 0; site[2] < Ns; ++site[2]) {
                for (site[3] = 0; site[3] < Ns; ++site[3]) {
                    double q = topological_charge_density(site);
                    Q_total += q;
                }
            }
        }
    }
    
    // Multiply by lattice spacing^4 (set to 1 if a=1)
    // For now, we use dimensionless lattice spacing a=1
    return Q_total;
}
