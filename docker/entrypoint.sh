#!/bin/bash

set -e

# Define workspace and build directoriess
WORKSPACE_DIR="/workspace"
BUILD_DIR="$WORKSPACE_DIR/build"

# Navigate to the build directory
cd "$BUILD_DIR"

# Configure the project with CMake
echo "Running CMake configuration..."
cmake "$WORKSPACE_DIR"

# Build the project
echo "Building the project..."
cmake --build . --config Release


