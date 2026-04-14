#include "../include/lattice.hpp"
#include "../include/gauge_ops.hpp"
#include <random>
#include <stdexcept>

namespace qcd {

LatticeLayout::LatticeLayout(const SimParams& p, int rank_, int nranks_)
    : Lx(p.Lx), Ly(p.Ly), Lz(p.Lz), Lt(p.Lt),
      rank(rank_), nranks(nranks_)
{
    if(p.Lz % nranks != 0)
        throw std::runtime_error("Lz must be divisible by nranks");
    Lz_local   = p.Lz / nranks;
    ghost_size = p.Lx * p.Ly * p.Lt;   // one Z-layer (full T extent)
    vol_local  = ghost_size * Lz_local;
    rank_prev  = (rank - 1 + nranks) % nranks;
    rank_next  = (rank + 1) % nranks;
}

GaugeField::GaugeField(const SimParams& p, int rank, int nranks)
    : params(p), layout(p, rank, nranks)
{
    int vol = layout.vol_local;
    int gs  = layout.ghost_size;
    int Lx  = layout.Lx, Ly = layout.Ly, Lz = layout.Lz;

    links.resize(NDIM * vol);
    momenta.resize(NDIM * vol);
    hopping.resize(NDIR * vol);

    for(int s=0; s<vol; s++) {
        int x, y, zl, t;
        layout.coords(s, x, y, zl, t);

        hopping[0*vol+s] = layout.site((x+1)%Lx, y, zl, t);           // +x (local)
        hopping[1*vol+s] = layout.site(x, (y+1)%Ly, zl, t);           // +y (local)

        if(zl < layout.Lz_local - 1)                                    // +z
            hopping[2*vol+s] = layout.site(x, y, zl+1, t);
        else
            hopping[2*vol+s] = vol + (x + Lx*(y + Ly*t));             // fwd ghost

        hopping[3*vol+s] = layout.site(x, y, zl, (t+1)%layout.Lt);    // +t (local, periodic)

        hopping[4*vol+s] = layout.site((x-1+Lx)%Lx, y, zl, t);        // -x (local)
        hopping[5*vol+s] = layout.site(x, (y-1+Ly)%Ly, zl, t);        // -y (local)

        if(zl > 0)                                                       // -z
            hopping[6*vol+s] = layout.site(x, y, zl-1, t);
        else
            hopping[6*vol+s] = vol + gs + (x + Lx*(y + Ly*t));        // bwd ghost

        hopping[7*vol+s] = layout.site(x, y, zl, (t-1+layout.Lt)%layout.Lt); // -t (local, periodic)
    }

    ghost_fwd.resize(NDIM * gs);
    ghost_bwd.resize(NDIM * gs);
}

void GaugeField::init_unity() {
    int vol = layout.vol_local;
    for(int mu=0; mu<NDIM; mu++)
        for(int s=0; s<vol; s++)
            link(mu, s) = SU3::identity();
}

void GaugeField::init_random(unsigned long long seed) {
    std::mt19937_64 rng(seed + (unsigned long long)layout.rank);
    int vol = layout.vol_local;
    for(int mu=0; mu<NDIM; mu++)
        for(int s=0; s<vol; s++) {
            link(mu, s) = random_su3_near_identity(rng, 0.5);
            reunitarize(link(mu, s));
        }
}

void GaugeField::exchange_halo() {
    int gs     = layout.ghost_size;   // = Lx*Ly*Lt
    int vol    = layout.vol_local;
    int Lx = layout.Lx, Ly = layout.Ly, Lt = layout.Lt;

    std::vector<SU3> send_next(NDIM * gs);
    std::vector<SU3> send_prev(NDIM * gs);
    for(int mu=0; mu<NDIM; mu++)
        for(int sp=0; sp<gs; sp++) {
            int x = sp % Lx, y = (sp/Lx) % Ly, t = sp / (Lx*Ly);
            send_next[mu*gs+sp] = link(mu, layout.site(x, y, layout.Lz_local-1, t));
            send_prev[mu*gs+sp] = link(mu, layout.site(x, y, 0, t));
        }

    if(layout.nranks == 1) {
        ghost_fwd = send_prev;
        ghost_bwd = send_next;
        return;
    }

    MPI_Status stat;
    MPI_Sendrecv(send_next.data(), (int)(NDIM*gs*sizeof(SU3)), MPI_BYTE,
                 layout.rank_next, 10,
                 ghost_bwd.data(), (int)(NDIM*gs*sizeof(SU3)), MPI_BYTE,
                 layout.rank_prev, 10, MPI_COMM_WORLD, &stat);
    MPI_Sendrecv(send_prev.data(), (int)(NDIM*gs*sizeof(SU3)), MPI_BYTE,
                 layout.rank_prev, 11,
                 ghost_fwd.data(), (int)(NDIM*gs*sizeof(SU3)), MPI_BYTE,
                 layout.rank_next, 11, MPI_COMM_WORLD, &stat);
}

const SU3& GaugeField::link_at(int mu, int s) const {
    int vol = layout.vol_local;
    int gs  = layout.ghost_size;
    if(s < vol)
        return links[mu*vol + s];
    else if(s < vol + gs)
        return ghost_fwd[mu*gs + (s - vol)];
    else
        return ghost_bwd[mu*gs + (s - vol - gs)];
}

// Staple sum: Sigma_{mu}(s) = sum_{nu != mu} [fwd_staple + bwd_staple]
SU3 GaugeField::staple(int mu, int site) const {
    int vol = layout.vol_local;
    SU3 sigma;
    for(int nu=0; nu<NDIM; nu++) {
        if(nu == mu) continue;
        int s_mu  = hopping[mu*vol + site];      // site + mu
        int s_nu  = hopping[nu*vol + site];      // site + nu
        int s_mnu = hopping[(nu+4)*vol + site];  // site - nu

        // Forward staple: U_nu(s+mu) * U_mu^dag(s+nu) * U_nu^dag(s)
        sigma += link_at(nu, s_mu) * link_at(mu, s_nu).dagger() * link_at(nu, site).dagger();

        // Backward staple: U_nu^dag(s+mu-nu) * U_mu^dag(s-nu) * U_nu(s-nu)
        int s_mu_mnu;
        if(s_mu < vol) {
            s_mu_mnu = hopping[(nu+4)*vol + s_mu];
        } else {
            // s_mu is in a z-ghost (mu==2 at z-boundary).
            // Ghost sp = x + Lx*(y + Ly*t), with full t extent.
            // Shifting in nu (x, y, t, or -z) stays within the same ghost layer
            // because t is fully periodic within the ghost.
            int gs   = layout.ghost_size;
            bool fwd = (s_mu < vol + gs);
            int base = fwd ? vol : (vol + gs);
            int sp   = s_mu - base;
            int gx   = sp % layout.Lx;
            int gy   = (sp / layout.Lx) % layout.Ly;
            int gt   = sp / (layout.Lx * layout.Ly);
            switch(nu) {
                case 0: gx = (gx - 1 + layout.Lx) % layout.Lx; break;
                case 1: gy = (gy - 1 + layout.Ly) % layout.Ly; break;
                case 3: gt = (gt - 1 + layout.Lt) % layout.Lt; break;
                // nu==2 (z) is excluded by nu!=mu and mu==2
            }
            s_mu_mnu = base + (gx + layout.Lx*(gy + layout.Ly*gt));
        }

        sigma += link_at(nu, s_mu_mnu).dagger() * link_at(mu, s_mnu).dagger() * link_at(nu, s_mnu);
    }
    return sigma;
}

} // namespace qcd
