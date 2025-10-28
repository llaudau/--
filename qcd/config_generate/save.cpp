<<<<<<< HEAD
#include "save.h"
#include <fstream>
#include <iostream>

void save_gauge_field_binary(const GaugeFieldType& field, const std::string& filepath) {
    std::ofstream file(filepath, std::ios::out | std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << filepath << std::endl;
        return;
    }

    file.write(reinterpret_cast<const char*>(&RANK), sizeof(int));
    
    for (int i = 0; i < RANK; ++i) {
        int dim_size = field.dimension(i);
        file.write(reinterpret_cast<const char*>(&dim_size), sizeof(int));
    }
    
    const size_t num_matrices = field.size(); 
    const size_t num_complex_elements = num_matrices * SU3Matrix::SizeAtCompileTime; 
    const size_t total_data_bytes = num_complex_elements * sizeof(ComplexD);
    file.write(reinterpret_cast<const char*>(field.data()), total_data_bytes);
    file.close();
    std::cout << "Successfully saved GaugeField (" << num_matrices << " SU3 matrices) to: " << filepath << std::endl;
}
=======
#include <fstream>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
// ... your original using statements and typedefs ...

void saveGaugeField(const GaugeFieldType& field, const std::string& filename) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }

    // 1. Write the rank (optional, but good for robust format)
    int rank = field.rank();
    file.write(reinterpret_cast<const char*>(&rank), sizeof(rank));

    // 2. Write the dimensions (for GaugeFieldType, this is an array of size RANK)
    const auto& dims = field.dimensions();
    for (int i = 0; i < rank; ++i) {
        // Use the Index type or a fixed-size type for safety
        Eigen::Tensor<SU3Matrix, RANK>::Index dim_val = dims[i];
        file.write(reinterpret_cast<const char*>(&dim_val), sizeof(dim_val));
    }
    
    // 3. Write the raw data
    // The size of the data is total number of SU3Matrix blocks * size of one SU3Matrix
    // Note: SU3Matrix is a Matrix<ComplexD, 3, 3>, so size is 3*3*sizeof(ComplexD)
    std::size_t total_size_bytes = field.size() * sizeof(SU3Matrix);
    
    // field.data() gives a pointer to SU3Matrix. 
    // We cast it to a char* for binary I/O.
    file.write(reinterpret_cast<const char*>(field.data()), total_size_bytes);

    file.close();
    std::cout << "Successfully saved GaugeField to " << filename << std::endl;
}

// ... Call it: saveGaugeField(my_gauge_field, "/path/to/your/directory/field.bin");
>>>>>>> 29cd4cc (local)
