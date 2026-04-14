#pragma once
#include "lattice.hpp"

namespace qcd {

double plaquette(const GaugeField& gf);
double topo_charge(const GaugeField& gf);
double hamiltonian_full(const GaugeField& gf);

// Clover energy density: E = -(1/V) sum_s sum_{mu<nu} Tr[F_mu_nu^2]  (Qlattice convention)
double clover_energy(const GaugeField& gf);

// Wilson loop W(r, t) averaged over the lattice, for spatial direction mu and temporal nu
double wilson_loop(const GaugeField& gf, int r, int t, int mu, int nu);

} // namespace qcd
