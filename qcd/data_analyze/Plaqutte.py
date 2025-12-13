import numpy as np
import matplotlib.pyplot as plt

basic_path="/home/khw/Documents/Git_repository/qcd/pure_gauge_contract/contracted_data/"
config_name="t10_s8_beta6.0/"
filepath = basic_path+config_name+'plaqutte_re.txt'

try:
    # np.loadtxt is highly efficient for reading numeric data from text files.
    # It automatically handles newlines and whitespace.
    data_array = np.loadtxt(filepath)
    
    print(f"Successfully read {data_array.size} elements.")
    print(f"The first 5 elements are: {data_array[:5]}")
    
except FileNotFoundError:
    print(f"Error: The file '{filepath}' was not found.")
except Exception as e:
    print(f"An error occurred during reading: {e}")
configs=np.arange(data_array.shape[0])


plt.plot(configs,data_array,marker='D',linestyle='-',linewidth='0.1',markersize =0.1)
plt.ylim(2.7,2.9)
plt.xlim(0,300)
plt.savefig("/home/khw/Documents/Git_repository/qcd/data_analyze/"+config_name+"plaqutte.png",dpi=600)