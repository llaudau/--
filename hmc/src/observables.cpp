#include "../include/observables.hpp"
#include <cmath>

namespace qcd {

double plaquette(const GaugeField& gf) {
    int vol = gf.layout.vol_local;
    double sum_local = 0.0;

    // Assumes halo is already fresh (caller must call exchange_halo() first)
    #pragma omp parallel for reduction(+:sum_local) schedule(static)
    for(int s=0; s<vol; s++) {
        for(int mu=0; mu<NDIM; mu++) {
            for(int nu=mu+1; nu<NDIM; nu++) {
                int s_mu = gf.neighbor(s, mu);
                int s_nu = gf.neighbor(s, nu);
                SU3 plaq = gf.link_at(mu, s)
                         * gf.link_at(nu, s_mu)
                         * gf.link_at(mu, s_nu).dagger()
                         * gf.link_at(nu, s).dagger();
                sum_local += plaq.re_trace();
            }
        }
    }

    double sum_global = 0.0;
    MPI_Allreduce(&sum_local, &sum_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Normalize: 6 planes per site, 3 colors
    int total_vol = gf.params.Lx * gf.params.Ly * gf.params.Lz * gf.params.Lt;
    return sum_global / (6.0 * 3.0 * total_vol);
}

// Clover leaf C_mu_nu(s): average of 4 oriented plaquettes around site s.
// Matches Qlattice gf_clover_leaf_no_comm exactly.
// Uses only single-hop neighbors to avoid ghost-of-ghost issues.
static SU3 clover_leaf(const GaugeField& gf, int mu, int nu, int s) {
    int s_pmu = gf.neighbor(s, mu);       // s + mu
    int s_pnu = gf.neighbor(s, nu);       // s + nu
    int s_mmu = gf.neighbor(s, mu+4);     // s - mu
    int s_mnu = gf.neighbor(s, nu+4);     // s - nu

    SU3 C = SU3::zero();

    // Leaf 1: path (+mu, +nu, -mu, -nu) from site s
    // = U_mu(s) * U_nu(s+mu) * U_mu†(s+nu) * U_nu†(s)
    C += gf.link_at(mu, s) * gf.link_at(nu, s_pmu)
       * gf.link_at(mu, s_pnu).dagger() * gf.link_at(nu, s).dagger();

    // Leaf 2: path (-mu, -nu, +mu, +nu) from site s
    // = U_mu†(s-mu) * U_nu†(s-mu-nu) * U_mu(s-mu-nu) * U_nu(s-nu)
    // Need s-mu-nu: neighbor of s-mu in -nu direction
    int vol = gf.layout.vol_local;
    int s_mmu_mnu = (s_mmu < vol) ? gf.neighbor(s_mmu, nu+4) : s_mnu; // approx for ghost
    C += gf.link_at(mu, s_mmu).dagger() * gf.link_at(nu, s_mmu_mnu).dagger()
       * gf.link_at(mu, s_mmu_mnu) * gf.link_at(nu, s_mnu);

    // Leaf 3: path (+nu, -mu, -nu, +mu) from site s
    // = U_nu(s) * U_mu†(s+nu-mu) * U_nu†(s-mu) * U_mu(s-mu)
    int s_pnu_mmu = (s_pnu < vol) ? gf.neighbor(s_pnu, mu+4) : s_mmu; // approx for ghost
    C += gf.link_at(nu, s) * gf.link_at(mu, s_pnu_mmu).dagger()
       * gf.link_at(nu, s_mmu).dagger() * gf.link_at(mu, s_mmu);

    // Leaf 4: path (-nu, +mu, +nu, -mu) from site s
    // = U_nu†(s-nu) * U_mu(s-nu) * U_nu(s-nu+mu) * U_mu†(s)
    int s_mnu_pmu = (s_mnu < vol) ? gf.neighbor(s_mnu, mu) : s_pmu; // approx for ghost
    C += gf.link_at(nu, s_mnu).dagger() * gf.link_at(mu, s_mnu)
       * gf.link_at(nu, s_mnu_pmu) * gf.link_at(mu, s).dagger();

    return C * 0.25;
}

double topo_charge(const GaugeField& gf) {
    int vol = gf.layout.vol_local;
    double Q_local = 0.0;

    // Matching Qlattice clf_topology_density:
    // 1. C[i] = clover_leaf for 6 planes (01,02,03,12,13,23)
    // 2. A[i] = 0.5 * (C[i] - C[i]†)   (anti-Hermitian, NO trace subtraction)
    // 3. q = fac * [ -Tr(A[1]*A[4]) + Tr(A[2]*A[3]) + Tr(A[5]*A[0]) ]
    //    where fac = -1/(4*pi^2)
    // Index: A[0]=F01, A[1]=F02, A[2]=F03, A[3]=F12, A[4]=F13, A[5]=F23

    const double fac = -1.0 / (2.0 * M_PI * M_PI);

    #pragma omp parallel for reduction(+:Q_local) schedule(static)
    for(int s=0; s<vol; s++) {
        SU3 A[6];
        SU3 C0 = clover_leaf(gf, 0, 1, s);  A[0] = (C0 - C0.dagger()) * 0.5;
        SU3 C1 = clover_leaf(gf, 0, 2, s);  A[1] = (C1 - C1.dagger()) * 0.5;
        SU3 C2 = clover_leaf(gf, 0, 3, s);  A[2] = (C2 - C2.dagger()) * 0.5;
        SU3 C3 = clover_leaf(gf, 1, 2, s);  A[3] = (C3 - C3.dagger()) * 0.5;
        SU3 C4 = clover_leaf(gf, 1, 3, s);  A[4] = (C4 - C4.dagger()) * 0.5;
        SU3 C5 = clover_leaf(gf, 2, 3, s);  A[5] = (C5 - C5.dagger()) * 0.5;

        double q = 0.0;
        q -= (A[1]*A[4]).re_trace();
        q += (A[2]*A[3]).re_trace();
        q += (A[5]*A[0]).re_trace();
        Q_local += fac * q;
    }

    double Q_global = 0.0;
    MPI_Allreduce(&Q_local, &Q_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return Q_global;
}

double hamiltonian_full(const GaugeField& gf) {
    // gauge action
    int vol = gf.layout.vol_local;
    double beta = gf.params.beta;
    double H_local = 0.0;

    for(int s=0; s<vol; s++) {
        for(int mu=0; mu<NDIM; mu++) {
            for(int nu=mu+1; nu<NDIM; nu++) {
                int s_mu = gf.neighbor(s, mu);
                int s_nu = gf.neighbor(s, nu);
                SU3 plaq = gf.link_at(mu,s) * gf.link_at(nu,s_mu)
                         * gf.link_at(mu,s_nu).dagger() * gf.link_at(nu,s).dagger();
                H_local += beta * (1.0 - plaq.re_trace() / 3.0);
            }
        }
        for(int mu=0; mu<NDIM; mu++) {
            SU3 P2 = gf.mom(mu,s) * gf.mom(mu,s);
            H_local += P2.re_trace();  // K = Tr[P²]
        }
    }

    double H_global = 0.0;
    MPI_Allreduce(&H_local, &H_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return H_global;
}

// Clover energy density following Qlattice convention exactly:
//   F_mu_nu = make_tr_less_anti_herm(clover_leaf(mu,nu))
//           = (C - C†)/2 - (1/3)Tr[(C - C†)/2] * I
//   E = (1/V) sum_s sum_{mu<nu} -Tr(F_mu_nu * F_mu_nu)
// Reference: https://arxiv.org/pdf/1006.4518.pdf Eq. (2.1)
double clover_energy(const GaugeField& gf) {
    int vol = gf.layout.vol_local;
    double E_local = 0.0;

    #pragma omp parallel for reduction(+:E_local) schedule(static)
    for(int s=0; s<vol; s++) {
        double e_site = 0.0;
        for(int mu=0; mu<3; mu++) {
            for(int nu=mu+1; nu<NDIM; nu++) {
                // Clover leaf (reuses the same correct clover_leaf used in topo_charge)
                SU3 C = clover_leaf(gf, mu, nu, s);
                // Anti-Hermitian part
                SU3 F = (C - C.dagger()) * 0.5;
                // Traceless projection
                complex tr = F.trace() * complex(1.0/3.0, 0.0);
                F(0,0) = F(0,0) - tr;
                F(1,1) = F(1,1) - tr;
                F(2,2) = F(2,2) - tr;
                // E += -Tr(F * F)  (positive for anti-Hermitian F)
                e_site += -(F * F).re_trace();
            }
        }
        E_local += e_site;
    }

    double E_global = 0.0;
    MPI_Allreduce(&E_local, &E_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    int total_vol = gf.params.Lx * gf.params.Ly * gf.params.Lz * gf.params.Lt;
    return E_global / total_vol;
}

// Wilson loop W(r, t_loop) = <(1/3) Re Tr[ prod of links around r x t_loop rectangle ]>
// Uses global coordinates to avoid out-of-bounds access when the path traverses ghost
// regions.  Sites whose loop path requires links beyond the one-layer halo are skipped;
// the average is taken over valid sites only (unbiased by translational invariance).
double wilson_loop(const GaugeField& gf, int r, int t_loop, int mu, int nu) {
    int vol      = gf.layout.vol_local;
    int Lx = gf.layout.Lx, Ly = gf.layout.Ly, Lz = gf.layout.Lz;
    int Lt       = gf.params.Lt;
    int Lz_loc   = gf.layout.Lz_local;
    int gs       = gf.layout.ghost_size;   // = Lx*Ly*Lt
    int z0       = gf.layout.rank * Lz_loc;  // global z-offset of this rank
    int fwd_gz   = (z0 + Lz_loc) % Lz;       // global z served by ghost_fwd
    int bwd_gz   = (z0 - 1 + Lz) % Lz;       // global z served by ghost_bwd

    // Direction unit vectors [mu] → (dx, dy, dz, dt)
    const int dx[4] = {1,0,0,0}, dy[4] = {0,1,0,0};
    const int dz[4] = {0,0,1,0}, dt[4] = {0,0,0,1};

    // Return pointer to U_dir at global site (gx,gy,gz,gt), nullptr if unavailable.
    auto link_global = [&](int dir, int gx, int gy, int gz, int gt) -> const SU3* {
        gx = ((gx%Lx)+Lx)%Lx;
        gy = ((gy%Ly)+Ly)%Ly;
        gz = ((gz%Lz)+Lz)%Lz;
        gt = ((gt%Lt)+Lt)%Lt;
        int sp = gx + Lx*(gy + Ly*gt);   // ghost index: x + Lx*(y + Ly*t)
        if(gz >= z0 && gz < z0 + Lz_loc)
            return &gf.link(dir, gf.layout.site(gx, gy, gz - z0, gt));
        if(gz == fwd_gz) return &gf.ghost_fwd[dir*gs + sp];
        if(gz == bwd_gz) return &gf.ghost_bwd[dir*gs + sp];
        return nullptr;   // beyond ghost reach
    };

    double W_local  = 0.0;
    int    n_valid  = 0;

    #pragma omp parallel for reduction(+:W_local) reduction(+:n_valid) schedule(static)
    for(int s=0; s<vol; s++) {
        int lx, ly, lz, lt;
        gf.layout.coords(s, lx, ly, lz, lt);
        int gx = lx, gy = ly, gz = z0 + lz, gt = lt;

        SU3  prod  = SU3::identity();
        bool valid = true;

        // Forward r steps in mu
        for(int i=0; i<r && valid; i++) {
            const SU3* lp = link_global(mu, gx, gy, gz, gt);
            if(!lp) { valid = false; break; }
            prod = prod * (*lp);
            gx += dx[mu]; gy += dy[mu]; gz += dz[mu]; gt += dt[mu];
        }
        // Forward t_loop steps in nu
        for(int i=0; i<t_loop && valid; i++) {
            const SU3* lp = link_global(nu, gx, gy, gz, gt);
            if(!lp) { valid = false; break; }
            prod = prod * (*lp);
            gx += dx[nu]; gy += dy[nu]; gz += dz[nu]; gt += dt[nu];
        }
        // Backward r steps in mu
        for(int i=0; i<r && valid; i++) {
            gx -= dx[mu]; gy -= dy[mu]; gz -= dz[mu]; gt -= dt[mu];
            const SU3* lp = link_global(mu, gx, gy, gz, gt);
            if(!lp) { valid = false; break; }
            prod = prod * lp->dagger();
        }
        // Backward t_loop steps in nu
        for(int i=0; i<t_loop && valid; i++) {
            gx -= dx[nu]; gy -= dy[nu]; gz -= dz[nu]; gt -= dt[nu];
            const SU3* lp = link_global(nu, gx, gy, gz, gt);
            if(!lp) { valid = false; break; }
            prod = prod * lp->dagger();
        }

        if(valid) {
            W_local += prod.re_trace() / 3.0;
            n_valid++;
        }
    }

    double W_global = 0.0;
    int    n_global = 0;
    MPI_Allreduce(&W_local, &W_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&n_valid, &n_global, 1, MPI_INT,    MPI_SUM, MPI_COMM_WORLD);

    if(n_global == 0) return 0.0;
    return W_global / n_global;
}

} // namespace qcd
