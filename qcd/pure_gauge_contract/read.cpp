#include "read.h"
#include <fstream>
#include <iostream>


LatticeData read_GaugeFieldData(const std::string& filepath){
    std::ifstream file(filepath, std::ios::in | std::ios::binary);
    LatticeData data;
    int rank_read;
    file.read(reinterpret_cast<char*>(&rank_read), sizeof(int));

    int* dim_ptr[5] = {&data.Lt, &data.Lx, &data.Ly, &data.Lz,&data.link_num};
    for (int i = 0; i < RANK; ++i) {
        int dim_size;
        file.read(reinterpret_cast<char*>(&dim_size), sizeof(int));
        *dim_ptr[i] = dim_size;
        // std::cout<<*dim_ptr[i]<<std::endl;
    }
    
    data.field.resize(data.Lt, data.Lx, data.Ly, data.Lz, data.link_num);
    // data.field.resize(8,4,4,4,2048);
    const size_t total_elements = data.field.size();
    const size_t total_data_bytes = total_elements * sizeof(SU3Matrix);
    file.read(reinterpret_cast<char*>(data.field.data()), total_data_bytes);
    file.close();
    return data ;
};


// LatticeData read_GaugeFieldData(const std::string& filepath){
//     std::ifstream file(filepath, std::ios::binary);
//     if (!file) {
//         throw std::runtime_error("Cannot open file: " + filepath);
//     }

//     LatticeData data;
//     int rank_in_file;
//     file.read(reinterpret_cast<char*>(&rank_in_file), sizeof(int));

//     // Safety check: ensure file matches code
//     if (rank_in_file != 5) { 
//         std::cerr << "Error: File rank " << rank_in_file << " does not match expected rank 5" << std::endl;
//         // Handle error...
//     }

//     // Read dimensions into a temporary array first to validate
//     int dims[5];
//     for (int i = 0; i < 5; ++i) {
//         file.read(reinterpret_cast<char*>(&dims[i]), sizeof(int));
//         if (dims[i] <= 0 || dims[i] > 1000) { // Basic sanity check
//             std::cerr << "Invalid dimension read: " << dims[i] << " at index " << i << std::endl;
//         }
//     }

//     data.Lt = dims[0]; data.Lx = dims[1]; data.Ly = dims[2]; data.Lz = dims[3]; data.link_num = dims[4];

//     // Allocate memory
//     data.field.resize(data.Lt, data.Lx, data.Ly, data.Lz, data.link_num);
    
//     // Calculate size correctly
//     const size_t total_elements = data.field.size();
//     const size_t total_data_bytes = total_elements * sizeof(SU3Matrix);

//     // Verify file has enough data remaining
//     file.seekg(0, std::ios::end);
//     size_t fileSize = file.tellg();
//     size_t currentPos = sizeof(int) + (5 * sizeof(int));
//     file.seekg(currentPos, std::ios::beg);

//     if (fileSize - currentPos < total_data_bytes) {
//         std::cerr << "Error: File is too small for the expected lattice size!" << std::endl;
//     }

//     file.read(reinterpret_cast<char*>(data.field.data()), total_data_bytes);
//     file.close();
    
//     return data;
// }