FROM python:3.13.0-slim-bookworm AS base

ENV CPPCHECK_VERSION=2.16.0

WORKDIR /tmp/cppcheck

RUN apt-get update && apt-get install -y \
    git \
    make \
    g++ \
    z3 \
    libz3-dev \
    libpcre3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && git clone --single-branch -b $CPPCHECK_VERSION https://github.com/danmar/cppcheck.git . \
    && make install USE_Z3=yes HAVE_RULES=yes MATCHCOMPILER=yes FILESDIR=/etc/cppcheck CXXFLAGS="-O2 -DNDEBUG -Wall -Wno-sign-compare -Wno-unused-function" \
    && echo -e "#!/bin/sh\ncppcheck --dump .\n misra.py \`find . -name '*.dump' | tr '\n' ' '\` 2> misra-report.txt" > /usr/bin/misra && chmod +x /usr/bin/misra

# Base image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"

# Install sudo 
RUN apt-get update && apt-get install -y \
    sudo && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create the docker group with the same GID
ARG USER
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG DOCKER_GID=984
RUN \
    groupadd -g $DOCKER_GID docker && \
    useradd -m -u 1000 -s /bin/bash ${USER} && \
    adduser ${USER} sudo && \
    usermod -aG docker ${USER} && \
    usermod -aG sudo ${USER} && \
    usermod -aG adm ${USER} && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers

USER ${USER}
WORKDIR /home/${USER}

# Install dependencies and clean up
RUN sudo apt-get update && sudo apt-get install -y \
    # Build tools and compilers
    build-essential \
    cmake \
    make \
    \
    # Version control and debugging tools
    git \
    gdb \
    clang-format \
    libgtest-dev \
    \
    # Network and file utilities
    wget \
    curl \
    unzip \
    \
    # Python environment
    python3 \
    python3-pip \
    \
    # Development libraries
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    libyaml-cpp-dev \
    \
    # Editors and connectivity
    vim \
    ssh \
    \
    # Containerization and multimedia
    docker.io \
    ffmpeg \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

# Ensure clang-format is in the PATH for vscode extension
ENV PATH="/usr/bin:${PATH}"

# Install a compatible GCC version (11 for LIBTorch with CUDA 11.8)
RUN sudo apt-get update && sudo apt-get install -y \
    g++-11 \
    gcc-11 \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

# Set GCC 11 as default
RUN sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# Create third_party directory for external libraries
RUN mkdir -p third_party

# Install cuDNN (ensure compatibility with CUDA 11.8)
RUN sudo apt-get update && sudo apt-get install -y \
    libcudnn8 \
    libcudnn8-dev \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

# Install PyTorch C++ library for CUDA 11.8 (compatible with GCC 11)
RUN curl -L https://download.pytorch.org/libtorch/nightly/cu118/libtorch-cxx11-abi-shared-with-deps-latest.zip -o libtorch.zip && \
    unzip libtorch.zip -d third_party/ && \
    rm libtorch.zip
ENV CMAKE_PREFIX_PATH="~/third_party/libtorch"

# Install cppcheck
RUN pip3 install --trusted-host pypi.org pygments \
    && sudo apt-get update && sudo apt-get install -y \
    z3 \
    libz3-dev \
    libpcre3-dev \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*
COPY --from=base /usr/bin/cppcheck /usr/bin/cppcheck-htmlreport /usr/bin/misra /usr/bin/*.py /usr/bin/
COPY --from=base /etc/cppcheck /etc/cppcheck

# Install Protobuf
RUN sudo apt-get update && sudo apt-get install -y \
    libprotobuf-dev \
    protobuf-compiler \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*
ENV CMAKE_PREFIX_PATH="/usr:/usr/local:${CMAKE_PREFIX_PATH}"

# Install Python packages
RUN pip3 install numpy matplotlib black tensorboard
