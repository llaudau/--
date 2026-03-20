#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <string>
#include <cmath>
#include "lattice.cuh"
namespace fs = std::filesystem;
using namespace qcdcuda;
int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "  HMC Autocorrelation Study" << std::endl;
    std::cout << "  Topological Charge & Plaquette" << std::endl;
    std::cout << "======================================" << std::endl;

    cudaSetDevice(0);

    // Lattice parameters
    int T = 16;
    int S = 8;
    int thermal_steps = 500;
    int n_samples = 10000;
    int ntraj = 10000;
    num_type trajectory_length = 0.1;
    int num_steps = 40;
    int smear_steps = 40;  // Number of gradient flow groups
    int flow_steps_per_group = 5;  // Each group does 20 flow steps
    num_type beta = 5.5;
    num_type epsilon = 0.01;   // gradient flow step size (t_max = 7*20*0.05 = 7.0 a^2, covering t0~5.5 at beta=6.5)

    std::cout << "\n--- Lattice Size: " << S << "x" << S << "x" << S << "x" << T << " ---" << std::endl;
    std::cout << "--- Beta = " << beta << ", Epsilon = " << epsilon << " ---" << std::endl;
    std::cout << "--- HMC trajectories: " << ntraj << ", Smear steps: " << smear_steps << " (each = " << flow_steps_per_group << " flow steps) ---" << std::endl;

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
    num_type* h_energy_data = (num_type*)malloc(data_size); // E(t) = 6.0 - plaquette

    // Allocate workspace for Wilson loops (Lt * Lx loops per configuration)
    int max_t = T;
    int max_x = S;
    int n_wilson_loops = max_t * max_x;
    size_t wilson_data_size = n_samples * n_wilson_loops * sizeof(num_type);
    num_type* h_wilson_data = (num_type*)malloc(wilson_data_size);

    std::cout << "\nThermalizing with HMC..." << std::endl;
    for (int i = 0; i < thermal_steps; i++) {
        lattice.update_1step(trajectory_length, num_steps);
        if (i % 20 == 0) {
            num_type plaq = lattice.calculate_plaquette()/6.0;
            std::cout << "  Thermal " << std::setw(4) << i << ": plaquette = " << std::fixed << std::setprecision(5) << plaq << std::endl;
        }
    }

    std::cout << "\nStarting production run..." << std::endl;
    std::cout << "  Traj | Q^2(0)    | P(0)    | Q^2(" << smear_steps-1 << ")    | P(" << smear_steps-1 << ")    " << std::endl;
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
            num_type Ene=lattice.Clover_ene();
            h_topo_data[idx] = Q;
            h_plaq_data[idx] = P/6.0;
            h_energy_data[idx] =Ene; // E(t) = sum_{mu<nu} [1 - (1/3)ReTr[P_munu]] / V

            // Calculate Wilson loops for unsmeared configuration (s=0)
            if (s == 0) {
                int wilson_idx_base = traj * n_wilson_loops;
                int wl_count = 0;
                for (int t_ext = 1; t_ext <= max_t; t_ext++) {
                    for (int x_ext = 1; x_ext <= max_x; x_ext++) {
                        // Calculate Wilson loop in x-t plane (mu=0 for x, nu=3 for t)
                        num_type wl = lattice.calculate_wilson_loop(x_ext, t_ext, 0, 3);
                        h_wilson_data[wilson_idx_base + wl_count] = wl;
                        wl_count++;
                    }
                }
            }

            lattice.gradient_flow(epsilon, flow_steps_per_group);
            
            // Calculate and show plaquette after each gradient flow group
            num_type P_after_flow = lattice.calculate_plaquette() / 6.0;
            if (traj % 1000 == 0) {
                std::cout << "    Flow step " << s << " (t=" << (s+1)*flow_steps_per_group << "): P = " << std::fixed << std::setprecision(5) << P_after_flow << std::endl;
            }
        }

        // Restore original (unsmeared) links for next update
        cudaMemcpy(lattice.d_links, d_links_copy, link_size, cudaMemcpyDeviceToDevice);

        // Print progress every 1000 trajectories
        if (traj % 1000 == 0) {
            int idx0 = traj * smear_steps + 0;
            int idx_last = traj * smear_steps + (smear_steps - 1);
            std::cout << "  " << std::setw(5) << traj 
                      << " | " << std::fixed << std::setprecision(4) << h_topo_data[idx0]
                      << " | " << std::setprecision(5) << h_plaq_data[idx0]
                      << " | " << std::setprecision(4) << h_topo_data[idx_last]
                      << " | " << std::setprecision(5) << h_plaq_data[idx_last] << std::endl;
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
    std::string wilson_path = results_dir + "/wilson_loop_data.txt";
    std::string energy_path = results_dir + "/energy_density_data.txt";
    std::string flow_times_path = results_dir + "/flow_times.txt";

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

    // Save energy density data: E(t) = 6.0 - plaquette, same layout as plaquette
    // outer loop = smear step, inner = trajectory
    std::ofstream energy_file(energy_path);
    for (int s = 0; s < smear_steps; s++) {
        for (int traj = 0; traj < n_samples; traj++) {
            int idx = traj * smear_steps + s;
            energy_file << std::setprecision(8) << h_energy_data[idx];
            if (!(traj == n_samples - 1 && s == smear_steps - 1)) {
                energy_file << "\n";
            }
        }
    }
    energy_file.close();
    std::cout << "  Saved: " << energy_path << " (" << n_samples * smear_steps << " values)" << std::endl;

    // Save flow times: one value per smear step (flow_time = s * flow_steps_per_group * epsilon)
    std::ofstream ft_file(flow_times_path);
    for (int s = 0; s < smear_steps; s++) {
        num_type t_flow = (num_type)s * flow_steps_per_group * epsilon;
        ft_file << std::setprecision(8) << t_flow;
        if (s < smear_steps - 1) ft_file << "\n";
    }
    ft_file.close();
    std::cout << "  Saved: " << flow_times_path << " (" << smear_steps << " flow times)" << std::endl;

    // Save Wilson loop data: format is (t_ext, x_ext, traj) flattened
    // outer loop = t_ext, then x_ext, inner = trajectory (matches wilson_loop_analysis.py)
    std::ofstream wilson_file(wilson_path);
    for (int t_ext = 1; t_ext <= max_t; t_ext++) {
        for (int x_ext = 1; x_ext <= max_x; x_ext++) {
            for (int traj = 0; traj < n_samples; traj++) {
                int wilson_idx = traj * n_wilson_loops + (t_ext - 1) * max_x + (x_ext - 1);
                wilson_file << std::setprecision(8) << h_wilson_data[wilson_idx];
                if (!(traj == n_samples - 1 && t_ext == max_t && x_ext == max_x)) {
                    wilson_file << "\n";
                }
            }
        }
    }
    wilson_file.close();
    std::cout << "  Saved: " << wilson_path << " (" << n_samples * n_wilson_loops << " values, " << max_t << "x" << max_x << " loops per config)" << std::endl;

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
        std::cout << "  <Q^2> = " << std::fixed << std::setprecision(4) << avg_Q << " +/- " << std::sqrt(var_Q) << std::endl;
        std::cout << "  <P> = " << std::fixed << std::setprecision(6) << avg_P << " +/- " << std::sqrt(var_P) << std::endl;
    }

    // Cleanup
    free(h_topo_data);
    free(h_plaq_data);
    free(h_energy_data);
    free(h_wilson_data);
    cudaFree(d_links_copy);

    std::cout << "\n======================================" << std::endl;
    std::cout << "  HMC Study Complete" << std::endl;
    std::cout << "======================================" << std::endl;

    return 0;
}
