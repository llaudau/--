#pragma once
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace qcd {

struct SimParams {
    // Lattice geometry
    int Lx = 16, Ly = 16, Lz = 16, Lt = 32;

    // HMC parameters
    double beta             = 6.0;
    double trajectory_length= 0.2;
    int    md_steps         = 40;
    int    thermal_steps    = 1000;
    int    n_samples        = 1000;
    int    print_interval   = 20;
    int    skip_steps       = 0;     // HMC updates between measurements (0 = measure every traj)

    // Smearing: "none", "flow"  (APE kept as library function but not used in run loop)
    std::string smear_method = "flow";

    // APE smearing parameters
    double ape_alpha        = 0.1;
    int    ape_steps        = 8;

    // Gradient flow (Wilson flow) parameters
    double flow_eps         = 0.01;
    double flow_t_max       = 2.0;

    // Parallelism: 8 MPI ranks per node (1 per NUMA domain), 8 OMP threads per rank
    int omp_threads         = 8;

    // RNG
    unsigned long long seed = 42;

    // Output
    std::string output_dir  = "";  // auto-generated if empty

    int volume() const { return Lx*Ly*Lz*Lt; }

    // Auto-generate output path: results/T{Lt}_S{Lx}/beta{beta}/{smear_method}
    std::string auto_output_dir() const {
        std::string sm = smear_method;
        if(sm == "none") sm = "raw";
        std::ostringstream oss;
        oss << "./results/T" << Lt << "_S" << Lx
            << "/beta" << std::fixed << std::setprecision(1) << beta
            << "/" << sm;
        return oss.str();
    }

    void print() const {
        std::cout << "=== SimParams ===\n";
        std::cout << "Lattice: " << Lx << "x" << Ly << "x" << Lz << "x" << Lt << "\n";
        std::cout << "Beta: " << beta << "\n";
        std::cout << "Traj len: " << trajectory_length << ", MD steps: " << md_steps << "\n";
        std::cout << "Thermal: " << thermal_steps << ", Samples: " << n_samples << ", Skip: " << skip_steps << "\n";
        std::cout << "Smearing: flow (eps=" << flow_eps << ", t_max=" << flow_t_max << ")\n";
        std::cout << "Wilson loop: r_max=" << Lx << " (=Ls), t_max=" << Lt << "\n";
        std::cout << "OMP threads: " << omp_threads << " (per node)\n";
        std::cout << "Seed: " << seed << "\n";
        std::cout << "Output: " << output_dir << "\n";
        std::cout << "=================\n";
    }
};

// Simple CLI parser — key=value or --key value style
inline SimParams parse_args(int argc, char** argv) {
    SimParams p;
    auto getval = [&](int i, const char* name) -> std::string {
        if(i >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
        return argv[i];
    };
    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if(arg=="--Ls")   { p.Lx = p.Ly = p.Lz = std::stoi(getval(++i,"Ls")); }
        else if(arg=="--Lx")   { p.Lx   = std::stoi(getval(++i,"Lx")); }
        else if(arg=="--Ly")   { p.Ly   = std::stoi(getval(++i,"Ly")); }
        else if(arg=="--Lz")   { p.Lz   = std::stoi(getval(++i,"Lz")); }
        else if(arg=="--Lt")   { p.Lt   = std::stoi(getval(++i,"Lt")); }
        else if(arg=="--beta") { p.beta = std::stod(getval(++i,"beta")); }
        else if(arg=="--traj") { p.trajectory_length = std::stod(getval(++i,"traj")); }
        else if(arg=="--md")   { p.md_steps = std::stoi(getval(++i,"md")); }
        else if(arg=="--thermal"){ p.thermal_steps = std::stoi(getval(++i,"thermal")); }
        else if(arg=="--samples"){ p.n_samples = std::stoi(getval(++i,"samples")); }
        else if(arg=="--print") { p.print_interval = std::stoi(getval(++i,"print")); }
        else if(arg=="--skip")  { p.skip_steps = std::stoi(getval(++i,"skip")); }
        else if(arg=="--smear") { p.smear_method = getval(++i,"smear"); }
        else if(arg=="--ape_alpha"){ p.ape_alpha = std::stod(getval(++i,"ape_alpha")); }
        else if(arg=="--ape_steps"){ p.ape_steps = std::stoi(getval(++i,"ape_steps")); }
        else if(arg=="--flow_eps"){ p.flow_eps = std::stod(getval(++i,"flow_eps")); }
        else if(arg=="--flow_t") { p.flow_t_max = std::stod(getval(++i,"flow_t")); }
        else if(arg=="--omp")   { p.omp_threads = std::stoi(getval(++i,"omp")); }
        else if(arg=="--seed")  { p.seed = std::stoull(getval(++i,"seed")); }
        else if(arg=="--out")   { p.output_dir = getval(++i,"out"); }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
        }
    }
    // Auto-generate output directory if not specified
    if(p.output_dir.empty())
        p.output_dir = p.auto_output_dir();
    return p;
}

} // namespace qcd
