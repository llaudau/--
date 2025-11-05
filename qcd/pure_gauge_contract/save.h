#pragma once
#include "lattice.h"
#include <string>

void save_contracted_data_binary(const Tensor<ComplexD,3>& field, const std::string& filepath);
void save_vector_to_text(const std::vector<double>& data, const std::string& filepath);