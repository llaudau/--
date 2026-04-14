#pragma once
#include "matrix.hpp"
#include "params.hpp"
#include <vector>
#include <mpi.h>

namespace qcd {

static const int NDIM = 4;
static const int NDIR = 8;  // forward + backward

struct LatticeLayout {
    int Lx, Ly, Lz, Lt;
    int Lz_local;
    int vol_local;   // Lx*Ly*Lz_local*Lt
    int ghost_size;  // Lx*Ly*Lt (one Z-layer, full T extent)

    int rank, nranks;
    int rank_prev, rank_next;

    LatticeLayout() = default;
    LatticeLayout(const SimParams& p, int rank_, int nranks_);

    // site index: x fastest, then y, then zl (local z), then t slowest
    int site(int x, int y, int zl, int t) const {
        return x + Lx*(y + Ly*(zl + Lz_local*t));
    }
    void coords(int s, int& x, int& y, int& zl, int& t) const {
        x  = s % Lx;  s /= Lx;
        y  = s % Ly;  s /= Ly;
        zl = s % Lz_local; s /= Lz_local;
        t  = s;
    }
};

class GaugeField {
public:
    SimParams     params;
    LatticeLayout layout;

    // links[mu*vol + site],  momenta[mu*vol + site]
    std::vector<SU3> links;
    std::vector<SU3> momenta;

    // hopping[dir*vol + site] = neighbor site index
    // dir: 0=+x 1=+y 2=+z 3=+t 4=-x 5=-y 6=-z 7=-t
    // Ghost index convention:
    //   [vol  .. vol+gs-1]    = +z ghost (from rank_next)
    //   [vol+gs.. vol+2*gs-1] = -z ghost (from rank_prev)
    std::vector<int> hopping;

    // ghost_fwd[mu*gs + sp] = links from rank_next (z+1 boundary), sp = x+Lx*(y+Ly*t)
    // ghost_bwd[mu*gs + sp] = links from rank_prev (z-1 boundary), sp = x+Lx*(y+Ly*t)
    std::vector<SU3> ghost_fwd;
    std::vector<SU3> ghost_bwd;

    GaugeField(const SimParams& p, int rank, int nranks);

    void init_unity();
    void init_random(unsigned long long seed);
    void exchange_halo();

    // Link accessor that handles ghost region transparently
    const SU3& link_at(int mu, int s) const;

    SU3& link(int mu, int site)             { return links[mu*layout.vol_local + site]; }
    const SU3& link(int mu, int site) const { return links[mu*layout.vol_local + site]; }

    SU3& mom(int mu, int site)             { return momenta[mu*layout.vol_local + site]; }
    const SU3& mom(int mu, int site) const { return momenta[mu*layout.vol_local + site]; }

    int neighbor(int site, int dir) const { return hopping[dir*layout.vol_local + site]; }

    // Compute staple sum for link (mu, site)
    SU3 staple(int mu, int site) const;
};

} // namespace qcd
