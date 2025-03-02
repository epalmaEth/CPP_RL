# Minimum CMake version required
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Project name
project(CPP_RL)

# Set C++ standard globally (preferably C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Enable CUDNN
set(CAFFE2_USE_CUDNN 1)

# Find the required packages
find_package(Torch REQUIRED)
find_package(Protobuf REQUIRED)

# Enable FetchContent module for external dependencies
include(FetchContent)

# Fetch and link yaml-cpp
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG "yaml-cpp-0.7.0"
)
FetchContent_MakeAvailable(yaml-cpp)

# Fetch and link tensorboard_logger
FetchContent_Declare(
    tensorboard_logger
    GIT_REPOSITORY      https://github.com/RustingSword/tensorboard_logger.git
    GIT_TAG             master
)
FetchContent_MakeAvailable(tensorboard_logger)

# Add include directories for your project
include_directories(include)

# Option to choose between building tests or the production code
option(BUILD_TESTS "Build unit tests instead of production code" OFF)

if(BUILD_TESTS)
    # Enable testing framework
    enable_testing()

    # Find and link GoogleTest
    find_package(GTest REQUIRED)

    target_link_libraries(unit_tests
        PRIVATE
        GTest::gtest
        GTest::gtest_main
    )

    # Collect all test source files
    file(GLOB_RECURSE TEST_SOURCES 
        "tests/storage/*.cpp")
    add_executable(unit_tests ${TEST_SOURCES})
    
    # Link third-party libraries to the test target
    target_link_libraries(unit_tests
        PRIVATE
        "${TORCH_LIBRARIES}"
        yaml-cpp::yaml-cpp
        tensorboard_logger
    )
    
    # Compiler options based on build type
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(unit_tests PRIVATE -O3 -march=native -flto)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(unit_tests PRIVATE -g -O0)
    endif()
    
    # Add tests for CTest
    add_test(NAME AllTests COMMAND unit_tests)

else()

    # Collect all source files
    file(GLOB 
        SOURCES 
        "src/main.cpp"
        "src/algorithms/*.cpp"
        "src/env/physics_based_envs/*.cpp"
        "src/modules/*.cpp"
        "src/modules/distributions/*.cpp"
        "src/modules/normalizers/*.cpp"
        "src/runners/*.cpp"
        "src/storage/*.cpp"
        "src/utils/*.cpp"
    )

    # Add the executable and all source files
    add_executable(cpp_rl ${SOURCES})

    # Apply compiler optimizations for Release builds
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(cpp_rl PRIVATE 
            -O3              # Optimization level 3
            -march=native     # Architecture-specific optimizations
            -flto             # Link Time Optimization
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(cpp_rl PRIVATE 
            -g                # Debugging symbols
            -O0               # No optimization
        )
    endif()

    # Link third-party libraries
    target_link_libraries(cpp_rl 
        PRIVATE 
        "${TORCH_LIBRARIES}"
        yaml-cpp::yaml-cpp
        tensorboard_logger
    )
endif()
