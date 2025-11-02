cmake_minimum_required(VERSION 3.10)

# 2. 命名项目
project(pure_G_contract CXX) # 指定项目名称和使用的语言 (C++ 是 CXX)

add_executable(pure_G_contract WilsonLoop.cpp read.cpp main.cpp save.cpp)

# 3. 设定 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
 
set_target_properties(pure_G_contract PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(pure_G_contract PUBLIC 
    ${CMAKE_SOURCE_DIR}/external/eigen-5.0.0 
)
