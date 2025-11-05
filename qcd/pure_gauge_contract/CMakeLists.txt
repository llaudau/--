cmake_minimum_required(VERSION 3.10)

project(pure_G_contract CXX)

find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    message(STATUS "OpenMP support enabled.")
endif()


add_executable(pure_G_contract WilsonLoop.cpp action.cpp read.cpp main.cpp save.cpp)

if (OpenMP_FOUND)
    target_link_libraries(pure_G_contract 
        PRIVATE OpenMP::OpenMP_CXX
    )
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
 
set_target_properties(pure_G_contract PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(pure_G_contract PUBLIC 
    ${CMAKE_SOURCE_DIR}/external/eigen-5.0.0 
)
