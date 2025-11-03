#include "save.h"
#include <fstream>
#include <iostream>

void save_contracted_data_binary(const Tensor<ComplexD,3>& shits, const std::string& filepath) {
    std::ofstream file(filepath, std::ios::out | std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << filepath << std::endl;
        return;
    }
    int RANK_shits=3;
    file.write(reinterpret_cast<const char*>(&RANK_shits), sizeof(int));
    
    int  total_complex_number=1;
    for (int i = 0; i < 3; ++i) {
        int dim_size = shits.dimension(i);
        total_complex_number*=dim_size;
        file.write(reinterpret_cast<const char*>(&dim_size), sizeof(int));
    }
    
    const size_t total_data_bytes = total_complex_number * sizeof(ComplexD);
    file.write(reinterpret_cast<const char*>(shits.data()), total_data_bytes);
    file.close();
    std::cout << "Successfully saved shits to: " << filepath << std::endl;
}