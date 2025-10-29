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
    }
    data.field.resize(data.Lt, data.Lx, data.Ly, data.Lz, data.link_num);
    const size_t total_elements = data.field.size();
    const size_t total_data_bytes = total_elements * sizeof(SU3Matrix);
    file.read(reinterpret_cast<char*>(data.field.data()), total_data_bytes);
    file.close();
    return data ;
};
