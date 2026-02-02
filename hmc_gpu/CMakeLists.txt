cmake_minimum_required(VERSION 3.18)
project(hmc_gpu LANGUAGES CXX CUDA)

# 1. Architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)

# 2. Separate Compilation - Important for linking different .cu files
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# 3. Define the executable
# Suggestion: rename .cpp to .cu if it includes matrix.cuh directly
add_executable(complex_test 
    tests/test_all.cu 
    lib/kernels/lattice.cu
    lib/kernels/phy_obse.cu
    lib/kernels/update_hmc.cu
    
)

# 4. Include Paths
target_include_directories(complex_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/structs
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 5. Compiler Flags
target_compile_options(complex_test PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)