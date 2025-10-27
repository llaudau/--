#include <fstream>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include <complex>
#include <Eigen/Dense>

using namespace Eigen;
using ComplexD = std::complex<double>;
using SU3Matrix = Matrix<ComplexD, 3, 3>;
using SU2Matrix =Matrix<ComplexD, 2, 2>;
using ComplexD = std::complex<double>;
const int RANK = 5;
using GaugeFieldType = Tensor<SU3Matrix, RANK>;

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