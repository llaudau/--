# Specify the minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# 1. Define the project and enable C++ (CXX) and CUDA as languages
project(GaugeQCD_PureGauge LANGUAGES CXX CUDA)

# 2. Set the C++ standard for all C++ and CUDA host code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 3. Define the CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 86)

# Apply the architecture flag and enable C++17 for the Device code
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -std=c++17 -O3")

# 4. Define all source files
set(SOURCE_FILES
    main.cpp
    update.cu
)

# 5. Create the executable target
add_executable(config_gen_gpu ${SOURCE_FILES})

# 6. Find the CUDA Toolkit
find_package(CUDA REQUIRED)

# 7. Link the executable against the CUDA runtime library
target_link_libraries(config_gen_gpu
    ${CUDA_LIBRARIES}
)

# Optional: Add the include directory for headers
target_include_directories(config_gen_gpu PUBLIC 
    .
)