#!/bin/bash

set -e

# Define directoriess
WORKSPACE_DIR="/workspace"
BUILD_DIR="$WORKSPACE_DIR/build"
SRC_DIR="$WORKSPACE_DIR/src"
INCLUDE_DIR="$WORKSPACE_DIR/include" 
PYTHON_DIR="$WORKSPACE_DIR/python"

# Code formatting
echo "Running clang-format..."
find "$SRC_DIR" "$INCLUDE_DIR" -type f \( -name '*.cpp' -o -name '*.h' \) | while read -r file; do
    # Create a temporary file for comparison
    temp_file=$(mktemp)
    
    # Generate the formatted output without modifying the file
    clang-format --style=Google "$file" > "$temp_file"
    
    # Check if the file needs formatting
    if ! diff -q "$file" "$temp_file" > /dev/null; then
        echo "Formatted: $file"
        # Apply the formatting changes
        mv "$temp_file" "$file"
    fi
done
echo "Running black..."
black "$PYTHON_DIR"

# Navigate to the build directory
cd "$BUILD_DIR"

# Configure the project with CMake
echo "Running CMake configuration..."
cmake "$WORKSPACE_DIR"

# Build the project
echo "Building the project..."
cmake --build .

