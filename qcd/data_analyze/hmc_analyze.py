import numpy as np
import os

def analyze_hmc_data(base_path, params_list):
    """
    Analyze HMC data for multiple parameter sets.
    Data is organized as:
        - Lines 0-9999: smear=0 (10000 samples)
        - Lines 10000-19999: smear=1 (10000 samples)
        - ...
        - Lines 70000-79999: smear=7 (10000 samples)
    
    Args:
        base_path: Base path to hmc_results folder
        params_list: List of (T, S, beta) tuples
    """
    results = []
    
    print("=" * 70)
    print("HMC Data Analysis - Smear=0 Condition")
    print("=" * 70)
    print(f"{'T':>4} {'S':>4} {'beta':>8} | {'<Plaq>':>10} {'sigma_P':>10} | {'<Q>':>10} {'sigma_Q':>10}")
    print("-" * 70)
    
    for T, S, beta in params_list:
        folder_name = f"t{T}_s{S}_beta{beta:.1f}"
        data_path = os.path.join(base_path, folder_name)
        
        # Read data files
        topo_file = os.path.join(data_path, "topo_charge_data.txt")
        plaq_file = os.path.join(data_path, "plaquette_data.txt")
        
        # Load data
        topo_data = np.loadtxt(topo_file)
        plaq_data = np.loadtxt(plaq_file)
        
        # Extract smear=0 data (first 10000 lines)
        topo_s0 = topo_data[:10000]
        plaq_s0 = plaq_data[:10000]
        
        # Calculate statistics
        avg_topo = np.mean(topo_s0)
        sigma_topo = np.std(topo_s0)
        avg_plaq = np.mean(plaq_s0)
        sigma_plaq = np.std(plaq_s0)
        
        print(f"{T:>4} {S:>4} {beta:>8.1f} | {avg_plaq:>10.6f} {sigma_plaq:>10.6f} | {avg_topo:>10.6f} {sigma_topo:>10.6f}")
        
        results.append({
            'T': T,
            'S': S,
            'beta': beta,
            'plaq_avg': avg_plaq,
            'plaq_sigma': sigma_plaq,
            'topo_avg': avg_topo,
            'topo_sigma': sigma_topo,
            'N_samples': len(plaq_s0)
        })
    
    print("=" * 70)
    
    return results

def save_results_to_file(results, output_file):
    """Save results to a text file."""
    with open(output_file, 'w') as f:
        f.write("HMC Analysis Results - Smear=0\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'T':>4} {'S':>4} {'beta':>8} | {'<P>':>10} {'sigma_P':>10} | {'<Q>':>10} {'sigma_Q':>10}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"{r['T']:>4} {r['S']:>4} {r['beta']:>8.1f} | {r['plaq_avg']:>10.6f} {r['plaq_sigma']:>10.6f} | {r['topo_avg']:>10.6f} {r['topo_sigma']:>10.6f}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total samples per config: {results[0]['N_samples']}\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    # Define parameter sets
    params_list = [
        (8, 4, 5.7),
        (8, 4, 6.0),
        (8, 4, 6.3),
        (10, 8, 5.7),
        (10, 8, 6.0),
        (10, 8, 6.3),
        (16, 10, 5.7),
        (16, 10, 6.0),
        (16, 10, 6.3),
    ]
    
    base_path = "/home/khw/Documents/Git_repository/hmc_gpu/hmc_results"
    output_file = "/home/khw/Documents/Git_repository/qcd/data_analyze/hmc_results_summary.txt"
    
    results = analyze_hmc_data(base_path, params_list)
    save_results_to_file(results, output_file)
