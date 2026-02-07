import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_smear_data(base_path, params_list):
    """
    Analyze HMC data for all smear steps.
    """
    all_results = {}
    
    for T, S, beta in params_list:
        folder_name = f"t{T}_s{S}_beta{beta:.1f}"
        data_path = os.path.join(base_path, folder_name)
        
        topo_file = os.path.join(data_path, "topo_charge_data.txt")
        plaq_file = os.path.join(data_path, "plaquette_data.txt")
        
        topo_data = np.loadtxt(topo_file)
        plaq_data = np.loadtxt(plaq_file)
        
        # Each smear step has 10000 samples
        samples_per_smear = 10000
        topo_by_smear = []
        plaq_by_smear = []
        
        for s in range(8):
            start = s * samples_per_smear
            end = (s + 1) * samples_per_smear
            topo_s = topo_data[start:end]
            plaq_s = plaq_data[start:end]
            topo_by_smear.append((np.mean(topo_s), np.std(topo_s)))
            plaq_by_smear.append((np.mean(plaq_s), np.std(plaq_s)))
        
        label = f"T={T}, S={S}"
        all_results[label] = {
            'plaq': plaq_by_smear,
            'topo': topo_by_smear,
            'beta': beta
        }
    
    return all_results

def plot_plaquette(all_results, output_file):
    """Plot plaquette vs smear steps."""
    plt.figure(figsize=(10, 7))
    
    smear_steps = np.arange(8)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    i = 0
    
    for label, data in all_results.items():
        plaq_data = data['plaq']
        means = [p[0] for p in plaq_data]
        sigmas = [p[1] for p in plaq_data]
        
        plt.errorbar(smear_steps+i*0.05, means, yerr=sigmas, 
                    label=label, marker=markers[i], 
                    color=colors[i], capsize=1, capthick=0.5,
                    linewidth=0.5, markersize=2)
        i += 1
    
    plt.xlabel('Smear Steps', fontsize=12)
    plt.ylabel('Plaquette', fontsize=12)
    plt.title('Plaquette vs Smear Steps (HMC) beta=6.0', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(smear_steps)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved: {output_file}")

def plot_topological_charge(all_results, output_file):
    """Plot topological charge vs smear steps."""
    plt.figure(figsize=(10, 7))
    
    smear_steps = np.arange(8)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    i = 0
    
    for label, data in all_results.items():
        topo_data = data['topo']
        means = [t[0] for t in topo_data]
        sigmas = [t[1] for t in topo_data]
        
        plt.errorbar(smear_steps+i*0.05, means, yerr=sigmas, 
                    label=label, marker=markers[i], 
                    color=colors[i], capsize=1, capthick=0.5,
                    linewidth=0.5, markersize=2)
        i += 1
    
    plt.xlabel('Smear Steps', fontsize=12)
    plt.ylabel('Topological Charge Q', fontsize=12)
    plt.title('Topological Charge vs Smear Steps (HMC) beta=6.0', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(smear_steps)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    params_list = [
        (8, 4, 6.0),
        (10, 8, 6.0),
        (16, 10, 6.0),
    ]
    
    base_path = "/home/khw/Documents/Git_repository/hmc_gpu/hmc_results"
    output_dir = "/home/khw/Documents/Git_repository/qcd/data_analyze/basic"
    
    all_results = analyze_smear_data(base_path, params_list)
    
    plot_plaquette(all_results, os.path.join(output_dir, "plaq_vs_smear.png"))
    plot_topological_charge(all_results, os.path.join(output_dir, "topo_vs_smear.png"))
    
    print("\nDone! Generated plots:")
    print(f"  - {output_dir}/plaq_vs_smear.png")
    print(f"  - {output_dir}/topo_vs_smear.png")
