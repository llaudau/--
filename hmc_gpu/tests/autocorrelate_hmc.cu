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
    std::cout << "  HMC Autocorrelation Study" << std::endl;
    std::cout << "  Topological Charge & Plaquette" << std::endl;
    std::cout << "======================================" << std::endl;

    cudaSetDevice(0);

    // Lattice parameters
    int T = 16;
    int S = 10;
    int thermal_steps = 500;
    int n_samples = 10000;
    int ntraj = 10000;
    num_type trajectory_length = 0.2;
    int num_steps = 40;
    int smear_steps = 8;
    num_type beta = 6.3;
    num_type alpha = 0.1;

    std::cout << "\n--- Lattice Size: " << S << "x" << S << "x" << S << "x" << T << " ---" << std::endl;
    std::cout << "--- Beta = " << beta << ", Alpha = " << alpha << " ---" << std::endl;
    std::cout << "--- HMC trajectories: " << ntraj << ", Smear steps: " << smear_steps << " ---" << std::endl;

    // Create lattice
    GaugeField lattice(S, S, S, T, beta);

    // Allocate workspace for smearing (copy of links)
    size_t link_size = lattice.params.volume * 4 * sizeof(Matrix<complex<num_type>, 3>);
    Matrix<complex<num_type>, 3>* d_links_copy;
    cudaMalloc(&d_links_copy, link_size);

    // Allocate workspace for Q and plaquette
    size_t data_size = n_samples * smear_steps * sizeof(num_type);
    num_type* h_topo_data = (num_type*)malloc(data_size);
    num_type* h_plaq_data = (num_type*)malloc(data_size);

    std::cout << "\nThermalizing with HMC..." << std::endl;
    for (int i = 0; i < thermal_steps; i++) {
        lattice.update_1step(trajectory_length, num_steps);
        if (i % 20 == 0) {
            num_type plaq = lattice.calculate_plaquette();
            std::cout << "  Thermal " << std::setw(4) << i << ": plaquette = " << std::fixed << std::setprecision(5) << plaq << std::endl;
        }
    }

    std::cout << "\nStarting production run..." << std::endl;
    std::cout << "  Traj | Q(0)    | P(0)    | Q(7)    | P(7)    " << std::endl;
    std::cout << "  -----+----------+----------+----------+----------" << std::endl;

    for (int traj = 0; traj < n_samples; traj++) {
        // HMC update
        lattice.update_1step(trajectory_length, num_steps);

        // Copy original links for smearing
        cudaMemcpy(d_links_copy, lattice.d_links, link_size, cudaMemcpyDeviceToDevice);

        // Calculate for each smear step
        for (int s = 0; s < smear_steps; s++) {
            int idx = traj * smear_steps + s;

            num_type Q = lattice.topo_charge();
            num_type P = lattice.calculate_plaquette();

            h_topo_data[idx] = Q;
            h_plaq_data[idx] = P;

            lattice.smear_links(alpha, 1);
        }

        // Restore original (unsmeared) links for next update
        cudaMemcpy(lattice.d_links, d_links_copy, link_size, cudaMemcpyDeviceToDevice);

        // Print progress every 1000 trajectories
        if (traj % 1000 == 0) {
            int idx0 = traj * smear_steps + 0;
            int idx7 = traj * smear_steps + 7;
            std::cout << "  " << std::setw(5) << traj 
                      << " | " << std::fixed << std::setprecision(4) << h_topo_data[idx0]
                      << " | " << std::setprecision(5) << h_plaq_data[idx0]
                      << " | " << std::setprecision(4) << h_topo_data[idx7]
                      << " | " << std::setprecision(5) << h_plaq_data[idx7] << std::endl;
        }
    }

    // Create results directory
    std::ostringstream dir_name;
    dir_name << "/home/khw/Documents/Git_repository/hmc_gpu/hmc_results/t" << T << "_s" << S << "_beta" << std::fixed << std::setprecision(1) << beta;
    std::string results_dir = dir_name.str();
    fs::create_directories(results_dir);

    std::cout << "\nSaving data to files..." << std::endl;

    std::string topo_path = results_dir + "/topo_charge_data.txt";
    std::string plaq_path = results_dir + "/plaquette_data.txt";

    // Save topological charge data
    std::ofstream topo_file(topo_path);
    for (int s = 0; s < smear_steps; s++) {
        for (int traj = 0; traj < n_samples; traj++) {
            int idx = traj * smear_steps + s;
            topo_file << std::setprecision(8) << h_topo_data[idx];
            if (!(traj == n_samples - 1 && s == smear_steps - 1)) {
                topo_file << "\n";
            }
        }
    }
    topo_file.close();
    std::cout << "  Saved: " << topo_path << " (" << n_samples * smear_steps << " values)" << std::endl;

    // Save plaquette data
    std::ofstream plaq_file(plaq_path);
    for (int s = 0; s < smear_steps; s++) {
        for (int traj = 0; traj < n_samples; traj++) {
            int idx = traj * smear_steps + s;
            plaq_file << std::setprecision(8) << h_plaq_data[idx];
            if (!(traj == n_samples - 1 && s == smear_steps - 1)) {
                plaq_file << "\n";
            }
        }
    }
    plaq_file.close();
    std::cout << "  Saved: " << plaq_path << " (" << n_samples * smear_steps << " values)" << std::endl;

    // Calculate statistics
    std::cout << "\n==============================================" << std::endl;
    std::cout << "  Statistics" << std::endl;
    std::cout << "==============================================" << std::endl;

    for (int s = 0; s < smear_steps; s++) {
        num_type avg_Q = 0, avg_P = 0, var_Q = 0, var_P = 0;
        for (int traj = 0; traj < n_samples; traj++) {
            int idx = traj * smear_steps + s;
            avg_Q += h_topo_data[idx];
            avg_P += h_plaq_data[idx];
        }
        avg_Q /= n_samples;
        avg_P /= n_samples;

        for (int traj = 0; traj < n_samples; traj++) {
            int idx = traj * smear_steps + s;
            var_Q += (h_topo_data[idx] - avg_Q) * (h_topo_data[idx] - avg_Q);
            var_P += (h_plaq_data[idx] - avg_P) * (h_plaq_data[idx] - avg_P);
        }
        var_Q /= n_samples;
        var_P /= n_samples;

        std::cout << "\nSmear step " << s << ":" << std::endl;
        std::cout << "  <Q> = " << std::fixed << std::setprecision(4) << avg_Q << " +/- " << std::sqrt(var_Q) << std::endl;
        std::cout << "  <P> = " << std::fixed << std::setprecision(6) << avg_P << " +/- " << std::sqrt(var_P) << std::endl;
    }

    // Cleanup
    free(h_topo_data);
    free(h_plaq_data);
    cudaFree(d_links_copy);

    std::cout << "\n======================================" << std::endl;
    std::cout << "  HMC Study Complete" << std::endl;
    std::cout << "======================================" << std::endl;

    return 0;
}
