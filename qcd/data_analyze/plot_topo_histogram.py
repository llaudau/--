import numpy as np
import matplotlib.pyplot as plt
import os

def plot_topo_histograms(base_path, params_list, output_dir):
    """
    Plot topological charge histograms for each smear step.
    """
    for T, S, beta in params_list:
        folder_name = f"t{T}_s{S}_beta{beta:.1f}"
        data_path = os.path.join(base_path, folder_name)
        
        topo_file = os.path.join(data_path, "topo_charge_data.txt")
        
        topo_data = np.loadtxt(topo_file)
        
        samples_per_smear = 10000
        
        for s in range(8):
            start = s * samples_per_smear
            end = (s + 1) * samples_per_smear
            topo_s = topo_data[start:end]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            bins = np.linspace(min(topo_s), max(topo_s), 30)
            
            counts, bin_edges, patches = ax.hist(topo_s, bins=bins, edgecolor='black', alpha=0.7)
            
            ax.set_xlabel('Topological Charge', fontsize=12)
            ax.set_ylabel('Number of Configurations', fontsize=12)
            ax.set_title(f'Topological Charge Distribution (Smear {s})\nT={T}, S={S}, $\\beta$={beta:.1f}', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            mean_val = np.mean(topo_s)
            std_val = np.std(topo_s)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.4f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1, label=f'Std: {std_val:.4f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1)
            ax.legend(fontsize=10)
            
            plt.tight_layout()
            
            output_file = os.path.join(output_dir, f"topo_hist_T{T}_S{S}_beta{beta:.1f}_smear{s}.png")
            plt.savefig(output_file, dpi=300)
            plt.close()
            print(f"Saved: {output_file}")

def plot_all_smears_combined(base_path, params_list, output_dir):
    """
    Plot all smear steps on one figure for comparison.
    """
    for T, S, beta in params_list:
        folder_name = f"t{T}_s{S}_beta{beta:.1f}"
        data_path = os.path.join(base_path, folder_name)
        
        topo_file = os.path.join(data_path, "topo_charge_data.txt")
        topo_data = np.loadtxt(topo_file)
        
        samples_per_smear = 10000
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for s in range(8):
            start = s * samples_per_smear
            end = (s + 1) * samples_per_smear
            topo_s = topo_data[start:end]
            
            bins = np.linspace(min(topo_s), max(topo_s), 30)
            
            axes[s].hist(topo_s, bins=bins, edgecolor='black', alpha=0.7)
            axes[s].set_xlabel('Topological Charge', fontsize=10)
            axes[s].set_ylabel('Number of Configs', fontsize=10)
            axes[s].set_title(f'Smear {s}', fontsize=12)
            axes[s].grid(True, alpha=0.3)
        
        plt.suptitle(f'Topological Charge Distribution (T={T}, S={S}, $\\beta$={beta:.1f})', fontsize=14)
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f"topo_hist_T{T}_S{S}_beta{beta:.1f}_all_smears.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Saved: {output_file}")

if __name__ == "__main__":
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
    output_dir = "/home/khw/Documents/Git_repository/qcd/data_analyze/result2"
    
    print("Generating individual histograms...")
    plot_topo_histograms(base_path, params_list, output_dir)
    
    print("\nGenerating combined plots...")
    plot_all_smears_combined(base_path, params_list, output_dir)
    
    print("\nDone!")
