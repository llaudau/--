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