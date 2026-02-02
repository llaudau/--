import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# basic_path="/home/khw/Documents/Git_repository/qcd/pure_gauge_contract/contracted_data/"
basic_path='/home/khw/Documents/Git_repository/hmc_gpu/results/'
T,S=10,8
autocorelate_sample_num=7
plt.figure(figsize=(18, 12))
for i in range(autocorelate_sample_num):
    betass=0.1*i+5.7
    config_name=f"t{T}_s{S}_beta{betass:.1f}/"
    filepath = basic_path+config_name+'plaquette_results.txt'
    
   
    try:

        data_array = np.loadtxt(filepath)
        
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred during reading: {e}")

    part=10000


    data_array=data_array[:part]
    def calculate_correlate_function(data):
        

        n=data.shape[0]
        Gamma=np.zeros(n)
        ave=np.average(data_array)
        var=np.var(data_array)
        for t in range(n-1):
            Gamma[t]+=var/n
            for i in range(n-t):
                Gamma[t]+=(data[i]-ave)*(data[t+i]-ave)/(n-t)
        return Gamma
    Gamma_Pleq=calculate_correlate_function(data_array)
    tao_cut=1/2+np.cumulative_sum(Gamma_Pleq)/Gamma_Pleq[0]
    t_cut=np.arange(part)

    plt.plot(t_cut,Gamma_Pleq)
    plt.xlabel('t')
    plt.ylabel('Gamma(t)')
    basic_save_path=f'/home/khw/Documents/Git_repository/qcd/data_analyze/t{T}_s{S}_hmc_beta_altcrlt_lgth/'
    plt.savefig(basic_save_path+"Gamma"+f"beta{betass:.2f}"+'.png',dpi=600)
    plt.clf()
    plt.plot(t_cut,tao_cut)
    plt.xlabel('W')
    plt.ylabel('tau (W)=1/2+Sum[rho (t),{t,1,W-1}]')
    plt.savefig(basic_save_path+"tao_cut"+f"beta{betass:.2f}"+'.png',dpi=600)
    np.save(basic_save_path+f"{betass:.2f}"+".npy",tao_cut)
    
    plt.clf()



