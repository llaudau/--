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
    std::cout << "  Large Lattice Topological Charge Test" << std::endl;
    std::cout << "======================================" << std::endl;

    cudaSetDevice(0);

    int T = 16;
    int S = 16;
    int thermal_steps = 200;
    int metro_steps = 5;
    num_type epsilon = 0.2;
    num_type beta = 6.0;

    std::cout << "\n--- Lattice Size: " << S << "x" << S << "x" << S << "x" << T << " ---" << std::endl;

    GaugeField lattice(S, S, S, T, beta);

    std::cout << "\nThermalizing..." << std::endl;
    for (int i = 0; i < thermal_steps; i++) {
        lattice.metropolis_update(epsilon, metro_steps);
        if (i % 20 == 0) {
            num_type plaq = lattice.calculate_plaquette();
            std::cout << "  Thermal " << std::setw(3) << i << ": plaquette = " << std::fixed << std::setprecision(5) << plaq << std::endl;
        }
    }

    std::cout << "\n--- Testing smearing with SMALL alpha = 0.02 ---" << std::endl;
    
    for (int iter = 0; iter < 10; iter++) {
        num_type Q_before = lattice.topo_charge();
        num_type plaq_before = lattice.calculate_plaquette();
        int nearest_before = (int)std::round(Q_before);
        num_type dist_before = std::abs(Q_before - nearest_before);
        
        std::cout << "Iter " << std::setw(2) << iter 
                  << ": Q = " << std::fixed << std::setprecision(4) << Q_before 
                  << " (sector " << nearest_before << ", dist=" << std::setprecision(3) << dist_before << ")"
                  << ", plaq=" << std::setprecision(5) << plaq_before 
                  << std::endl;
        
        lattice.smear_links(0.02, 1);
    }

    std::cout << "\n======================================" << std::endl;
    std::cout << "  Test Complete" << std::endl;
    std::cout << "======================================" << std::endl;

    return 0;
}
