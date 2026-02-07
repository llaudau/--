import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_autocorr(data, max_tau=500):
    """
    Calculate autocorrelation function Gamma(t).
    Gamma(t) = (1/(N-t)) * sum_{i=1}^{N-t} (O_i - <O>)(O_{i+t} - <O>)
    """
    n = len(data)
    ave = np.mean(data)
    var = np.var(data)
    
    max_tau = min(max_tau, n // 2)
    Gamma = np.zeros(max_tau)
    
    for t in range(max_tau):
        gamma_t = 0.0
        for i in range(n - t):
            gamma_t += (data[i] - ave) * (data[i + t] - ave)
        Gamma[t] = gamma_t / (n - t)
    
    return Gamma, var

def analyze_autocorr_data(base_path, params_list):
    """
    Analyze autocorrelation for all parameter sets.
    Use all samples step by step.
    """
    results = {}
    
    for T, S, beta in params_list:
        folder_name = f"t{T}_s{S}_beta{beta:.1f}"
        data_path = os.path.join(base_path, folder_name)
        
        topo_file = os.path.join(data_path, "topo_charge_data.txt")
        plaq_file = os.path.join(data_path, "plaquette_data.txt")
        
        topo_data = np.loadtxt(topo_file)
        plaq_data = np.loadtxt(plaq_file)
        
        samples_per_smear = 10000
        
        for s in range(8):
            start = s * samples_per_smear
            end = start + samples_per_smear
            
            plaq_s = plaq_data[start:end]
            topo_s = topo_data[start:end]
            
            Gamma_plaq, var_plaq = calculate_autocorr(plaq_s)
            Gamma_topo, var_topo = calculate_autocorr(topo_s)
            
            data_key = f"T{T}_S{S}_beta{beta:.1f}_smear{s}"
            results[data_key] = {
                'T': T,
                'S': S,
                'beta': beta,
                'smear': s,
                'plaq_Gamma': Gamma_plaq,
                'plaq_var': var_plaq,
                'topo_Gamma': Gamma_topo,
                'topo_var': var_topo,
                'plaq_data': plaq_s,
                'topo_data': topo_s
            }
    
    return results

def plot_autocorr_by_lattice(results, output_dir):
    """
    Plot autocorrelation for different beta values on the same lattice.
    """
    betas = [5.7, 6.0, 6.3]
    lattice_sizes = [(8, 4), (10, 8), (16, 10)]
    colors = {'5.7': '#e41a1c', '6.0': '#377eb8', '6.3': '#4daf4a'}
    
    for T, S in lattice_sizes:
        plt.figure(figsize=(12, 8))
        
        for beta in betas:
            key = f"T{T}_S{S}_beta{beta:.1f}_smear0"
            if key in results:
                data = results[key]
                Gamma = data['plaq_Gamma']
                var = data['plaq_var']
                rho = Gamma / var if var > 0 else Gamma
                rho[0] = 1.0
                
                t = np.arange(len(rho))
                plt.plot(t[1:201], rho[1:201], label=f'$\\beta={beta}$', 
                        color=colors[f'{beta:.1f}'], linewidth=1.5, alpha=0.8)
        
        plt.xlabel('$t$ (MD time)', fontsize=12)
        plt.ylabel(r'$\rho(t)$', fontsize=12)
        plt.title(f'Plaquette Autocorrelation - $T={T}, S={S}$', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'plaq_autocorr_T{T}_S{S}.png'), dpi=300)
        plt.close()
        print(f"Saved: plaq_autocorr_T{T}_S{S}.png")
        
        plt.figure(figsize=(12, 8))
        
        for beta in betas:
            key = f"T{T}_S{S}_beta{beta:.1f}_smear0"
            if key in results:
                data = results[key]
                Gamma = data['topo_Gamma']
                var = data['topo_var']
                rho = Gamma / var if var > 0 else Gamma
                rho[0] = 1.0
                
                t = np.arange(len(rho))
                plt.plot(t[1:201], rho[1:201], label=f'$\\beta={beta}$', 
                        color=colors[f'{beta:.1f}'], linewidth=1.5, alpha=0.8)
        
        plt.xlabel('$t$ (MD time)', fontsize=12)
        plt.ylabel(r'$\rho(t)$', fontsize=12)
        plt.title(f'Topological Charge Autocorrelation - $T={T}, S={S}$', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'topo_autocorr_T{T}_S{S}.png'), dpi=300)
        plt.close()
        print(f"Saved: topo_autocorr_T{T}_S{S}.png")

def plot_autocorr_with_smear(results, output_dir):
    """
    Plot autocorrelation for different smear steps.
    """
    betas = [5.7, 6.0, 6.3]
    lattice_sizes = [(8, 4), (10, 8), (16, 10)]
    smear_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                   '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for T, S in lattice_sizes:
        for beta in betas:
            plt.figure(figsize=(12, 8))
            
            for s in range(8):
                key = f"T{T}_S{S}_beta{beta:.1f}_smear{s}"
                if key in results:
                    data = results[key]
                    Gamma = data['plaq_Gamma']
                    var = data['plaq_var']
                    rho = Gamma / var if var > 0 else Gamma
                    rho[0] = 1.0
                    
                    t = np.arange(len(rho))
                    plt.plot(t[1:201], rho[1:201], label=f'$n_{{smear}}={s}$', 
                            color=smear_colors[s], linewidth=1.2, alpha=0.8)
            
            plt.xlabel('$t$ (MD time)', fontsize=12)
            plt.ylabel(r'$\rho(t)$', fontsize=12)
            plt.title(f'Plaquette Autocorr - $T={T}, S={S}, \\beta={beta}$', fontsize=14)
            plt.legend(fontsize=9, ncol=2)
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'plaq_smear_T{T}_S{S}_beta{beta:.1f}.png'), dpi=300)
            plt.close()
            
            plt.figure(figsize=(12, 8))
            
            for s in range(8):
                key = f"T{T}_S{S}_beta{beta:.1f}_smear{s}"
                if key in results:
                    data = results[key]
                    Gamma = data['topo_Gamma']
                    var = data['topo_var']
                    rho = Gamma / var if var > 0 else Gamma
                    rho[0] = 1.0
                    
                    t = np.arange(len(rho))
                    plt.plot(t[1:201], rho[1:201], label=f'$n_{{smear}}={s}$', 
                            color=smear_colors[s], linewidth=1.2, alpha=0.8)
            
            plt.xlabel('$t$ (MD time)', fontsize=12)
            plt.ylabel(r'$\rho(t)$', fontsize=12)
            plt.title(f'Topo Autocorr - $T={T}, S={S}, \\beta={beta}$', fontsize=14)
            plt.legend(fontsize=9, ncol=2)
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'topo_smear_T{T}_S{S}_beta{beta:.1f}.png'), dpi=300)
            plt.close()
            
        print(f"Saved autocorr plots for T={T}, S={S}, beta={beta}")

def save_integrated_autocorr(results, params_list, output_file):
    """
    Calculate and save integrated autocorrelation time.
    tau_int = 0.5 + sum_{t=1}^{N/2} rho(t)
    """
    with open(output_file, 'w') as f:
        f.write("Integrated Autocorrelation Time Analysis\n")
        f.write("=" * 70 + "\n")
        f.write("tau_int = 0.5 + sum_{t=1}^{N/2} rho(t)\n\n")
        
        for T, S, beta in params_list:
            f.write(f"\nLattice: $T={T}, S={S}, \\beta={beta:.1f}$\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Smear':<10}{'tau_int(Plaq)':<18}{'tau_int(Topo)':<18}\n")
            f.write("-" * 50 + "\n")
            
            for s in range(8):
                key = f"T{T}_S{S}_beta{beta:.1f}_smear{s}"
                if key in results:
                    data = results[key]
                    
                    Gamma_p = data['plaq_Gamma']
                    var_p = data['plaq_var']
                    rho_p = Gamma_p / var_p if var_p > 0 else Gamma_p
                    rho_p[0] = 1.0
                    
                    Gamma_t = data['topo_Gamma']
                    var_t = data['topo_var']
                    rho_t = Gamma_t / var_t if var_t > 0 else Gamma_t
                    rho_t[0] = 1.0
                    
                    N = len(rho_p)
                    half = N // 2
                    
                    tau_int_plaq = 0.5 + np.sum(rho_p[1:half])
                    tau_int_topo = 0.5 + np.sum(rho_t[1:half])
                    
                    f.write(f"{s:<10}{tau_int_plaq:<18.4f}{tau_int_topo:<18.4f}\n")
            
            f.write("\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"Saved integrated autocorr times to: {output_file}")

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
    output_dir = "/home/khw/Documents/Git_repository/qcd/data_analyze/result1"
    
    print("Analyzing autocorrelation functions...")
    results = analyze_autocorr_data(base_path, params_list)
    
    print("\nGenerating plots by lattice size...")
    plot_autocorr_by_lattice(results, output_dir)
    
    print("\nGenerating plots with different smear steps...")
    plot_autocorr_with_smear(results, output_dir)
    
    print("\nSaving integrated autocorrelation times...")
    save_integrated_autocorr(results, params_list, 
                           os.path.join(output_dir, "integrated_autocorr.txt"))
    
    print("\nDone! All results saved to:", output_dir)
