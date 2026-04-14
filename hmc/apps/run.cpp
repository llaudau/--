#include "../include/hmc.hpp"
#include "../include/observables.hpp"
#include "../include/gradient_flow.hpp"
#include "../include/io.hpp"
#include "../include/params.hpp"
#include <mpi.h>
#include <omp.h>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <string>

using namespace qcd;

struct FlowCtx {
    std::vector<double> t_vals, Q_vals, E_vals;
};

void flow_cb(double t, double Q, double E, void* ctx) {
    FlowCtx* fc = (FlowCtx*)ctx;
    fc->t_vals.push_back(t);
    fc->Q_vals.push_back(Q);
    fc->E_vals.push_back(E);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    SimParams p = parse_args(argc, argv);
    omp_set_num_threads(p.omp_threads);

    if(rank == 0) {
        std::cout << "=== HMC SU(3) MPI/CPU (Tianhe) ===\n";
        std::cout << "MPI ranks: " << nranks << ", OMP threads: " << p.omp_threads << "\n";
        p.print();
    }

    std::mt19937_64 rng(p.seed + (unsigned long long)rank);

    GaugeField gf(p, rank, nranks);
    gf.init_random(p.seed);
    gf.exchange_halo();

    // Only rank 0 creates directories and clears old CSVs, then all ranks sync.
    // Avoids race conditions on Lustre/GPFS parallel filesystems (Tianhe).
    if(rank == 0) ensure_dir(p.output_dir);
    MPI_Barrier(MPI_COMM_WORLD);

    std::string plaq_path  = p.output_dir + "/Plaq.csv";
    std::string wl_path    = p.output_dir + "/WS_loop.csv";
    std::string smear_path = p.output_dir + "/smear.csv";

    if(rank == 0) {
        std::filesystem::remove(plaq_path);
        std::filesystem::remove(wl_path);
        std::filesystem::remove(smear_path);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ---- Thermalization ----
    if(rank == 0) std::cout << "\n--- Thermalizing (" << p.thermal_steps << " steps) ---\n";
    int total_accept = 0;
    for(int i=0; i<p.thermal_steps; i++) {
        gf.exchange_halo();
        total_accept += hmc_step(gf, rng, p.trajectory_length, p.md_steps);
        if((i+1) % p.print_interval == 0) {
            gf.exchange_halo();
            double plaq = plaquette(gf);
            if(rank == 0) {
                double rate = (double)total_accept / (i+1) * 100.0;
                std::cout << "  Thermal " << std::setw(5) << i+1
                          << "  plaq=" << std::fixed << std::setprecision(5) << plaq
                          << "  acc=" << std::setprecision(1) << rate << "%\n";
            }
        }
    }

    // ---- Production ----
    if(rank == 0) std::cout << "\n--- Production (" << p.n_samples << " samples) ---\n";
    total_accept = 0;

    for(int n=0; n<p.n_samples; n++) {
        // Skip steps: extra HMC updates without measurement
        for(int sk=0; sk<p.skip_steps; sk++) {
            gf.exchange_halo();
            hmc_step(gf, rng, p.trajectory_length, p.md_steps);
        }

        gf.exchange_halo();
        total_accept += hmc_step(gf, rng, p.trajectory_length, p.md_steps);
        gf.exchange_halo();

        // --- Plaq.csv: raw plaquette on unsmeared config ---
        double plaq = plaquette(gf);
        if(rank == 0)
            append_csv(plaq_path, "traj,plaq", {(double)n, plaq});

        // --- WS_loop.csv: Wilson loops on unsmeared config ---
        // Average W(r,t) over 3 spatial directions (mu=0,1,2) with temporal direction nu=3
        for(int r=1; r<=p.Lx; r++) {
            for(int t=1; t<=p.Lt; t++) {
                double W = 0.0;
                for(int mu=0; mu<3; mu++)
                    W += wilson_loop(gf, r, t, mu, 3);
                W /= 3.0;
                if(rank == 0)
                    append_csv(wl_path, "traj,r,t,W", {(double)n, (double)r, (double)t, W});
            }
        }

        // --- smear.csv: Wilson flow on copy, record Q and E at each step ---
        FlowCtx fc;
        run_flow(gf, p.flow_eps, p.flow_t_max, flow_cb, &fc);
        if(rank == 0) {
            for(size_t i=0; i<fc.t_vals.size(); i++)
                append_csv(smear_path, "traj,t_flow,Q,E",
                           {(double)n, fc.t_vals[i], fc.Q_vals[i], fc.E_vals[i]});
        }

        if(rank == 0 && (n+1) % p.print_interval == 0) {
            double rate = (double)total_accept / (n+1) * 100.0;
            double Q_final = fc.Q_vals.empty() ? 0.0 : fc.Q_vals.back();
            std::cout << "  Sample " << std::setw(5) << n+1
                      << "  plaq=" << std::fixed << std::setprecision(5) << plaq
                      << "  Q=" << std::setprecision(3) << Q_final
                      << "  acc=" << std::setprecision(1) << rate << "%\n";
        }
    }

    if(rank == 0)
        std::cout << "\n=== Done. Results in " << p.output_dir << " ===\n";

    MPI_Finalize();
    return 0;
}
