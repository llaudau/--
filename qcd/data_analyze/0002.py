import numpy as np
import struct
import os
import matplotlib.pyplot as plt

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
            final_array = flat_array.reshape(dimensions)
            
            return final_array

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# # Example Usage (Replace 'data.bin' with your actual file path):
array_3d = read_3d_complex_array('/Users/wangkehe/Git_repository/qcd/pure_gauge_contract/contracted_data/test2.bin')
t=np.arange(16)
plt.scatter(t,np.average(array_3d.real,axis=0)[:,3])
plt.show()