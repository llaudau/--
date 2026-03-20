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
# add_executable(autocorrelate_study
#     tests/autocorrelate_study.cu
#     lib/kernels/lattice.cu
#     lib/kernels/phy_obse.cu
#     lib/kernels/update_hmc.cu
#     lib/kernels/update_Metro.cu
# )

add_executable(testdyn
    tests/autocorrelate_hmc.cu
    lib/kernels/lattice.cu
    lib/kernels/gauge_ape.cu
    lib/kernels/gauge_wilson_flow.cu
    lib/kernels/dirac.cu
    lib/kernels/cg.cu
    lib/kernels/update_hmc.cu
    # lib/kernels/update_hmc_dynferm.cu
    lib/kernels/phy_obse.cu
    # lib/kernels/update_Metro.cu
)

# 4. Include Paths
# target_include_directories(autocorrelate_study PRIVATE 
#     ${CMAKE_CURRENT_SOURCE_DIR}/include/structs
#     ${CMAKE_CURRENT_SOURCE_DIR}
# )

target_include_directories(testdyn PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/structs
    ${CMAKE_CURRENT_SOURCE_DIR}/include/linalg
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 5. Compiler Flags
# target_compile_options(autocorrelate_study PRIVATE 
#     $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
#     $<$<COMPILE_LANGUAGE:CXX>:-Wall>
# )

target_compile_options(testdyn PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)