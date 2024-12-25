# cmake_minimum_required specifies the minimum version of CMake required
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Project name
project(CPP_RL)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Set debug build
set(CMAKE_BUILD_TYPE Debug)

# Specify the path to libtorch directly
set(Torch_DIR /third_party/libtorch/share/cmake/Torch)

# Find the Torch package
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Set the path to Eigen headers (Eigen is header-only)
include_directories(/third_party/eigen-master/Eigen)

# Add your source files here
add_executable(cpp_rl src/main.cpp)

# Link third-party libraries
target_link_libraries(cpp_rl "${TORCH_LIBRARIES}")

set_property(TARGET cpp_rl PROPERTY CXX_STANDARD 17)



