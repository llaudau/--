#include "lattice.h"
#include <string>

struct LatticeData{
    int Lt;
    int Lx;
    int Ly;
    int Lz;
    int link_num;
    GaugeFieldType field;
};
// Read from a certain directory
LatticeData read_GaugeFieldData(const std::string& filepath);
