#include "save.h"
#include <fstream>
#include <iostream>
#include <iomanip>

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
    std::cout<<total_complex_number<<std::endl;
    const size_t total_data_bytes = total_complex_number * sizeof(ComplexD);
    file.write(reinterpret_cast<const char*>(shits.data()), total_data_bytes);
    file.close();
    std::cout << "Successfully saved shits to: " << filepath << std::endl;
}


void save_vector_to_text(const std::vector<double>& data, const std::string& filepath) {
    // 1. Open the output file stream
    std::ofstream output_file(filepath);

    // 2. Check if the file was successfully opened
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
        return;
    }
    output_file << std::fixed << std::setprecision(10);
    // 3. Write each double to the file, followed by a newline
    for (const double& value : data) {
        output_file << value << "\n"; 
        // std::cout << std::fixed << std::setprecision(6) << value << std::endl;
    }

    // 4. Close the file and confirm
    output_file.close();
    std::cout << "Successfully saved " << data.size() << " elements to: " << filepath << std::endl;
}