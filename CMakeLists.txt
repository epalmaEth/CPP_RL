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

# Set the path to the packages
set(CMAKE_PREFIX_PATH "/third_party/libtorch" ${CMAKE_PREFIX_PATH})

# Find the Torch package
find_package(Torch REQUIRED)

# Enable FetchContent module
include(FetchContent)
# Declare yaml-cpp dependency
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG "yaml-cpp-0.7.0"  # Replace with actual tag or commit hash
)
FetchContent_MakeAvailable(yaml-cpp)

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
target_link_libraries(cpp_rl 
    PUBLIC 
    "${TORCH_LIBRARIES}"
    yaml-cpp::yaml-cpp
)



