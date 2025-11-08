cmake_minimum_required(VERSION 3.10)
project(config_gen CXX)
find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    message(STATUS "OpenMP support enabled.")
endif()
add_executable(config_gen main.cpp update_pure_gauge.cpp save.cpp lattice.cpp)
if (OpenMP_FOUND)
    target_link_libraries(config_gen 
        PRIVATE OpenMP::OpenMP_CXX
    )
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_target_properties(config_gen PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_include_directories(config_gen PUBLIC 
    ${CMAKE_SOURCE_DIR}/external/eigen-5.0.0 
)
