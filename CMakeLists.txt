# cmake_minimum_required specifies the minimum version of CMake required
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Project name
project(CPP_RL)

# Set C++ standard globally (preferably C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Set debug build
set(CMAKE_BUILD_TYPE Debug)

# Specify the path to libtorch directly
set(Torch_DIR /third_party/libtorch/share/cmake/Torch)

# Find the Torch package
find_package(Torch REQUIRED)

# Set the path to Eigen headers (Eigen is header-only)
include_directories(/third_party/eigen-master/Eigen)

# Add include directories for your project
include_directories(include)

# Collect all source files in src/env/ directory (including pendulum.cpp)
file(GLOB 
    SOURCES 
    "src/main.cpp"
    "src/env/*.cpp"
    "src/modules/*.cpp"
)

# Add the executable and all source files
add_executable(cpp_rl ${SOURCES})

# Link third-party libraries
target_link_libraries(cpp_rl "${TORCH_LIBRARIES}")




