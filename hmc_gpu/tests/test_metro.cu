#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <string>
#include <cmath>
#include "lattice.cuh"
namespace fs = std::filesystem;

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "  Metropolis Update Test for SU(3)" << std::endl;
    std::cout << "  With Topological Charge Measurement" << std::endl;
    std::cout << "======================================" << std::endl;

    cudaSetDevice(0);

    int T = 8;
    int S = 4;
    int thermal_steps = 100;
    int metro_steps = 5;
    int ntraj = 200;
    int sample_interval = 20;
    num_type epsilon = 0.2;
    num_type beta = 6.0;

    std::cout << "\n--- Lattice Size: " << S << "x" << S << "x" << S << "x" << T << " ---" << std::endl;
    std::cout << "--- Beta = " << beta << " ---" << std::endl;

    GaugeField lattice(S, S, S, T, beta);

    num_type initial_plaq = lattice.calculate_plaquette();
    std::cout << "Initial plaquette: " << std::fixed << std::setprecision(6) << initial_plaq << std::endl;

    std::cout << "Running " << thermal_steps << " thermalization steps..." << std::endl;
    for (int i = 0; i < thermal_steps; i++) {
        num_type acc_rate = lattice.metropolis_update(epsilon, metro_steps);
        if (i % 20 == 0) {
            num_type plaq = lattice.calculate_plaquette();
            num_type topo = lattice.topo_charge();
            std::cout << "  Thermal " << std::setw(3) << i << ": plaquette = " << plaq
                      << ", topo_charge = " << std::fixed << std::setprecision(4) << topo
                      << ", acceptance = " << acc_rate << std::endl;
        }
    }

    num_type thermalized_plaq = lattice.calculate_plaquette();
    num_type thermalized_topo = lattice.topo_charge();
    std::cout << "Thermalized plaquette: " << std::fixed << std::setprecision(6) << thermalized_plaq << std::endl;
    std::cout << "Thermalized topo_charge: " << std::fixed << std::setprecision(4) << thermalized_topo << std::endl;

    std::vector<num_type> plaquette_history;
    std::vector<num_type> topo_history;
    std::vector<num_type> acceptance_history;

    std::cout << "\nRunning " << ntraj << " production trajectories..." << std::endl;
    std::cout << "Sampling every " << sample_interval << " steps..." << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << std::setw(8) << "Traj" 
              << std::setw(14) << "Plaquette" 
              << std::setw(14) << "Topo_Charge"
              << std::setw(12) << "Accept" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

    for (int i = 0; i < ntraj; i++) {
        num_type acc_rate = lattice.metropolis_update(epsilon, metro_steps);
        
        if (i % sample_interval == 0) {
            num_type plaq = lattice.calculate_plaquette();
            num_type topo = lattice.topo_charge();

            plaquette_history.push_back(plaq);
            topo_history.push_back(topo);
            acceptance_history.push_back(acc_rate);

            std::cout << std::setw(8) << i 
                      << std::fixed << std::setprecision(6) << std::setw(14) << plaq
                      << std::fixed << std::setprecision(4) << std::setw(14) << topo
                      << std::setw(12) << acc_rate << std::endl;
        }
    }

    std::cout << "--------------------------------------------" << std::endl;

    int n_samples = plaquette_history.size();
    num_type avg_plaq = 0;
    num_type avg_acc = 0;
    num_type avg_topo = 0;
    for (int i = 0; i < n_samples; i++) {
        avg_plaq += plaquette_history[i];
        avg_topo += topo_history[i];
        avg_acc += acceptance_history[i];
    }
    avg_plaq /= n_samples;
    avg_topo /= n_samples;
    avg_acc /= n_samples;

    num_type variance_plaq = 0;
    num_type variance_topo = 0;
    for (int i = 0; i < n_samples; i++) {
        variance_plaq += (plaquette_history[i] - avg_plaq) * (plaquette_history[i] - avg_plaq);
        variance_topo += (topo_history[i] - avg_topo) * (topo_history[i] - avg_topo);
    }
    variance_plaq /= n_samples;
    variance_topo /= n_samples;

    num_type topo_sq = 0;
    for (int i = 0; i < n_samples; i++) {
        topo_sq += topo_history[i] * topo_history[i];
    }
    topo_sq /= n_samples;

    std::cout << "\n==============================================" << std::endl;
    std::cout << "  Summary for Beta = " << beta << std::endl;
    std::cout << "  Lattice Size: " << S << "x" << S << "x" << S << "x" << T << std::endl;
    std::cout << "  Total samples: " << n_samples << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "Average plaquette:     " << std::fixed << std::setprecision(6) << avg_plaq << std::endl;
    std::cout << "Plaquette variance:    " << variance_plaq << std::endl;
    std::cout << "Average topo_charge:   " << std::fixed << std::setprecision(4) << avg_topo << std::endl;
    std::cout << "Topo_charge variance:  " << variance_topo << std::endl;
    std::cout << "<Q^2>:                 " << std::fixed << std::setprecision(4) << topo_sq << std::endl;
    std::cout << "Average acceptance:     " << avg_acc << std::endl;
    std::cout << "Expected plaquette:    ~" << (3.0 - beta / (3.0 * 2.0)) << std::endl;

    std::cout << "\n======================================" << std::endl;
    std::cout << "  Test Complete" << std::endl;
    std::cout << "======================================" << std::endl;

    return 0;
}
