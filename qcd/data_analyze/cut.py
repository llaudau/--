import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


T,S=8,4
s=1.5
autocorelate_sample_num=7

basic_path=f'/home/khw/Documents/Git_repository/qcd/data_analyze/t{T}_s{S}_beta_altcrlt_lgth/'

def EW(data, s):
    n = data.shape[0]
    W = np.arange(n)

    with np.errstate(divide='ignore', invalid='ignore'):
        data=abs(data) ###!!! some points in beta=5.70 have very low like data[i]=-0.8 and lead to overflow
        exponent = -W / (data*s)
        term1 = np.exp(exponent)
        term2 = 2 * (W / n)**0.5
        result = term1 + term2
        
    # Identify where values are infinite (overflow) or Not-a-Number (invalid)
    overflow_indices = np.where(np.isinf(result))[0]

    
    return result, overflow_indices
# plt.figure(figsize=(18, 12))
for i in range(autocorelate_sample_num):
    betass=0.1*i+5.7
    data=np.load(basic_path+f"{betass:.2f}"+".npy")
    data=data[:3000]

    W=np.arange(data.shape[0])
    EW0,bad1=EW(data,s)
    min_index = np.argmin(EW0)
    plt.plot(W,EW0,label=f'min_EW, W={min_index}')
    plt.title(r"$E(w)=e^{-W/(S \tau)}+2 \sqrt{W/N}$")
    plt.legend()
    plt.savefig(basic_path+'EW/'+f'EW_{betass:.2f}'+".png",dpi=300)
    plt.clf()
    
    