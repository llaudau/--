#pragma once
#include "lattice.hpp"

namespace qcd {

void flow_step(GaugeField& gf, double eps);
void run_flow(const GaugeField& gf_orig, double eps, double t_max,
              void (*callback)(double t, double Q, double E, void* ctx), void* ctx);

} // namespace qcd
