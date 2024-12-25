#!/bin/bash

set -e

# Define workspace and build directories
LIBTORCH_PATH="/third_party/libtorch"
WORKSPACE_DIR="/workspace"
BUILD_DIR="$WORKSPACE_DIR/build"

# Check if the build directory contains a recent build file
if [ -f "$BUILD_DIR/cpp_rl" -a $(stat -c %Y "$BUILD_DIR/cpp_rl") -gt $(date --date='10 minutes ago' +%s) ]; then
  echo "Recent build file found, skipping build process."
else
    # Create and clear the build directory
    echo "Setting up build directory..."
    rm -rf "$BUILD_DIR"/*

    # Navigate to the build directory
    cd "$BUILD_DIR"

    # Configure the project with CMake
    echo "Running CMake configuration..."
    cmake -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" "$WORKSPACE_DIR"

    # Build the project
    echo "Building the project..."
    cmake --build . --config Release
fi


