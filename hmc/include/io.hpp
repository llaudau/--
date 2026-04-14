#pragma once
#include "hmc.hpp"
#include "observables.hpp"
#include "gradient_flow.hpp"

namespace qcd {

void write_data(const std::string& path, const std::vector<double>& data);
void append_csv(const std::string& path, const std::string& header,
                const std::vector<double>& row);
void ensure_dir(const std::string& dir);

} // namespace qcd
