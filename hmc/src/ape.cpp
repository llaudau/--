#include "../include/ape.hpp"
#include "../include/gauge_ops.hpp"

namespace qcd {

void smear_ape(GaugeField& gf, double alpha, int n_iter) {
    int vol = gf.layout.vol_local;
    double coeff = alpha / 6.0;
    double selfcoeff = 1.0 - alpha;
    std::vector<SU3> links_new(NDIM * vol);

    for(int iter = 0; iter < n_iter; iter++) {
        gf.exchange_halo();

        #pragma omp parallel for schedule(static)
        for(int s = 0; s < vol; s++) {
            for(int mu = 0; mu < NDIM; mu++) {
                // staple() returns Σ used in HMC force: Tr[U*Σ†] = plaquette
                // APE needs Σ†: U_new = (1-α)*U + (α/6)*Σ†
                SU3 sigma_dag = gf.staple(mu, s).dagger();
                links_new[mu*vol + s] = gf.link(mu, s) * selfcoeff + sigma_dag * coeff;
            }
        }

        // Copy back and reunitarize
        for(int mu = 0; mu < NDIM; mu++)
            for(int s = 0; s < vol; s++) {
                gf.link(mu, s) = links_new[mu*vol + s];
                reunitarize(gf.link(mu, s));
            }
    }
}

} // namespace qcd
