#!/bin/bash

set -e

# Define directories
WORKSPACE_DIR="/home/$(whoami)/CPP_RL"
BUILD_DIR="$WORKSPACE_DIR/build"
SRC_DIR="$WORKSPACE_DIR/src"
INCLUDE_DIR="$WORKSPACE_DIR/include"
PYTHON_DIR="$WORKSPACE_DIR/python"

# Default build type to Release if no argument is provided
BUILD_TYPE="Release"

# Parse command-line arguments for build type
while [[ $# -gt 0 ]]; do
  case $1 in
    --debug)
      BUILD_TYPE="Debug"
      shift
      ;;
    --release)
      BUILD_TYPE="Release"
      shift
      ;;
    *)
      echo "Usage: $0 [--debug|--release]"
      exit 1
      ;;
  esac
done

# Code formatting
echo "Running clang-format..."
find "$SRC_DIR" "$INCLUDE_DIR" -type f \( -name '*.cpp' -o -name '*.h' \) | while read -r file; do
    # Create a temporary file for comparison
    temp_file=$(mktemp)
    
    # Generate the formatted output without modifying the file
    clang-format --style=file "$file" > "$temp_file"
    
    # Check if the file needs formatting
    if ! diff -q "$file" "$temp_file" > /dev/null; then
        echo "Formatted: $file"
        # Apply the formatting changes
        mv "$temp_file" "$file"
    else
        # Remove the temporary file
        rm "$temp_file"
    fi
done
echo "Running black..."
black "$PYTHON_DIR"

# cppcheck
echo "Running cppcheck..."
cppcheck --enable=all --std=c++17 --language=c++ \
         --check-level=exhaustive --suppress=missingIncludeSystem \
         --force --error-exitcode=1 \
         -I "$INCLUDE_DIR" "$SRC_DIR" "$INCLUDE_DIR"

# Ensure build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir -p "$BUILD_DIR"
fi

# Navigate to the build directory
cd "$BUILD_DIR"

# Configure the project with CMake
echo "Running CMake configuration..."
cmake "$WORKSPACE_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

# Build the project
echo "Building the project..."
cmake --build . -- -j $(nproc)
