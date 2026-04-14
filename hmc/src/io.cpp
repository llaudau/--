#include "../include/lattice.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <stdexcept>

namespace qcd {

// Write a column of data to a text file (one value per line)
void write_data(const std::string& path, const std::vector<double>& data) {
    std::ofstream f(path);
    if(!f) throw std::runtime_error("Cannot open: " + path);
    f << std::setprecision(10);
    for(double v : data) f << v << "\n";
}

// Append a single row to a CSV file (creates file + header if new)
void append_csv(const std::string& path, const std::string& header,
                const std::vector<double>& row)
{
    bool newfile = !std::filesystem::exists(path);
    std::ofstream f(path, std::ios::app);
    if(!f) throw std::runtime_error("Cannot open: " + path);
    if(newfile) f << header << "\n";
    f << std::setprecision(10);
    for(size_t i=0; i<row.size(); i++) {
        f << row[i];
        if(i+1 < row.size()) f << ",";
    }
    f << "\n";
}

// Create output directory (ok if exists)
void ensure_dir(const std::string& dir) {
    std::filesystem::create_directories(dir);
}

} // namespace qcd
