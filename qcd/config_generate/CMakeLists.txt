cmake_minimum_required(VERSION 3.10)

# 2. 命名项目
project(config_gen CXX) # 指定项目名称和使用的语言 (C++ 是 CXX)

add_executable(config_gen main.cpp update_pure_gauge.cpp save.cpp)

# 3. 设定 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
 

target_include_directories(config_gen PUBLIC 
    ${CMAKE_SOURCE_DIR}/external/eigen-5.0.0 
)
