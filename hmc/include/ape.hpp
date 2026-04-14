#pragma once
#include "lattice.hpp"

namespace qcd {

// APE smearing: U_new = Proj_SU3[(1-alpha)*U + (alpha/6)*staple]
// Modifies links in-place. Caller should save/restore if original config is needed.
void smear_ape(GaugeField& gf, double alpha, int n_iter);

} // namespace qcd
