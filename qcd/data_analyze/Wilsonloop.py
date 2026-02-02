import numpy as np
import struct
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

def read_3d_complex_array(filepath, metadata_format='<iiii'):
    """
    Reads a 3D array of complex double numbers from a binary file.

    :param filepath: The path to the binary data file.
    :param metadata_format: Struct format string for the 4 integers in the header (e.g., '<iiii' for 4 little-endian ints).
    :return: A 3D NumPy array of complex128, or None if reading fails.
    """
    try:
        # --- 1. Read Metadata Header (4 integers) ---
        metadata_size = struct.calcsize(metadata_format)
        
        with open(filepath, 'rb') as f:
            # Read the 16 bytes (4 integers) for the header
            header_bytes = f.read(metadata_size)
            if len(header_bytes) < metadata_size:
                print("Error: File is too small to contain the full header.")
                return None
            
            # Unpack the 4 integers: (dim_count, dim0, dim1, dim2)
            # Example: (3, 500, 16, 10)
            metadata = struct.unpack(metadata_format, header_bytes)
            
            dim_count = metadata[0]
            print(dim_count)
            if dim_count != 3:
                print(f"Warning: Expected 3 dimensions, found {dim_count}.")
                return None

            # The dimension sizes (shape)
            dimensions = metadata[1:]
            
            print(f"Metadata Read: {dim_count} dimensions with shape {dimensions}")

            # --- 2. Read Complex Double Data ---
            
            # The file pointer 'f' is already positioned right after the header.
            # 'c16' means complex numbers made of two 64-bit (8-byte) floats (complex double).
            # We must specify the total count of elements remaining in the file.
            
            # Calculate total expected data points
            total_elements = np.prod(dimensions)
            
            # Use numpy.fromfile to efficiently read the rest of the file
            # Note: We must pass 'f' (the file handle), not the filepath
            flat_array = np.fromfile(f, dtype=np.complex128, count=total_elements)

            # Check if the data size matches the expected size
            if flat_array.size != total_elements:
                print(f"Error: Expected {total_elements} elements, but read {flat_array.size}. Data may be truncated.")
                return None
            
            # --- 3. Reshape the 1D data into the 3D array ---
            final_array = flat_array.reshape(dimensions,order="F")
            
            return final_array

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# # Example Usage (Replace 'data.bin' with your actual file path):
basic_path='/home/khw/Documents/Git_repository/qcd/pure_gauge_contract/contracted_data/'


# config_name='t10_s8_beta6.0/'
config_name='t8_s4_beta6.0/'
# config_name='t8_s4_beta6.3/'
# T=10
# S=8
T=8
S=4
array_3d = read_3d_complex_array(basic_path+config_name+'Wilsonloop.bin')
print(array_3d.shape)

def jackknife(array):
    configs=array.shape[0]
    arraysum=np.sum(array,axis=0)
    print(arraysum.shape)
    arraysum=np.tile(arraysum,(configs,1,1))
    return (arraysum-array)/configs

def statistics(array):
    configs=array.shape[0]
    average=np.average(array,axis=0)
    variance=(np.var(array,axis=0))**(1/2)*(configs-1)
    return average, variance

def cov(array):
    configs=array.shape[0]
    cov_matrix=np.cov(array,ddof=0)*configs
    return cov_matrix

def Exp(x,A,m):
    return A*np.exp(-m*x)

def fit(array):
    
    configs=array.shape[0]
    Amplitude=np.zeros((configs,S*S))
    Energy=np.zeros((configs,S*S))
    t=np.arange(T)
    for x in range(S*S):
        covof_array=cov(array[:,:,x].T)
        for config in range(configs):
            out=curve_fit(Exp,t[1:],array[config,1:,x],sigma=covof_array[1:,1:])
            Amplitude[config,x]=out[0][0]
            Energy[config,x]=out[0][1]
    return Amplitude,Energy
qantiq=jackknife(array_3d.real)
amp,en=fit(qantiq)
print(en.shape)

t=np.arange(T)
x0=np.zeros(S*S)
for i in range(S*S):
    x0[i]=((i%4)*(i%4)+(i//4)*(i//4))**(1/2)

print(t)
aveen,varen=statistics(en)
plt.errorbar(x0,aveen,varen,ms=2,fmt='o',elinewidth=1,capsize=2,label="y=A/r+Br")
plt.xlabel("x/a")
plt.legend()
plt.savefig("/home/khw/Documents/Git_repository/qcd/data_analyze/"+config_name+"potential.png",dpi=600)


# ave,var=statistics(qantiq)

# plt.figure(figsize=(10, 8))
# for x in range(4):
#     plt.errorbar(t,ave[:,x],var[:,x],ms=2,fmt='o',elinewidth=1,capsize=2,label='x = '+str(x)+'a')
   
# plt.ylabel("Wilsonloop")
# plt.xlabel("t/a")
# plt.legend()
# plt.savefig("/home/khw/Documents/Git_repository/qcd/data_analyze/"+config_name+"wilsonloop.png",dpi=300)



















def renormalization(en):
    def p(x,A,B):
        return A/x+B*x

    configs=en.shape[0]
    x=np.zeros(6)
    y=np.zeros((configs,6))
    
    order=0
    for i in range(8):
        if i%4==0:
            continue
        x[order]=((i%4)**2+9*(i//4))**(1/2)
        y[:,order]=en[:,i]
        order+=1
    
    aveen,varen=statistics(y)
    cov_of_y=cov(y.T)
    
    A=np.zeros(configs)
    B=np.zeros(configs)
    for config in range(configs):
        out=curve_fit(p,x,y[config,:],sigma=cov_of_y)
        A[config]=out[0][0]
        B[config]=out[0][1]
    
    testnum=200
    testx=np.linspace(x[0],x[-1],testnum)
    testy=np.zeros((configs,testnum))
    for i in range(configs):
        testy[i,:]=p(testx,A[i],B[i])
    avey,vary=statistics(testy)

    plt.errorbar(testx,avey,vary,ms=2,fmt='o',elinewidth=1,capsize=2,label="fit: y=A/r+Br")

    plt.errorbar(x,aveen,varen,ms=2,fmt='o',elinewidth=1,capsize=2)
    plt.xlabel("x/a")
    plt.legend()
    plt.savefig("/home/khw/Documents/Git_repository/qcd/data_analyze/"+config_name+"potential.png",dpi=600)

    return
# renormalization(en)