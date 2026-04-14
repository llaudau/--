#pragma once
#include "lattice.hpp"
#include <random>

namespace qcd {

void refresh_momenta(GaugeField& gf, std::mt19937_64& rng);
void update_force(GaugeField& gf, double eps);
void update_links(GaugeField& gf, double eps);
void leapfrog(GaugeField& gf, double traj_len, int md_steps);
double hamiltonian(const GaugeField& gf);
int hmc_step(GaugeField& gf, std::mt19937_64& rng, double traj_len, int md_steps);

} // namespace qcd
