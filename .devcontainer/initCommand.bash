#!/bin/bash

set -ex

# Build the custom devcontainer image
DOCKER_BUILDKIT=1 docker build  \
                         . \
                         -f .devcontainer/Dockerfile \
                         -t devcontainer:cpp_rl \
                         --build-arg USER=$(whoami)

