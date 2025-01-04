# Base image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

USER root

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"

# Add user with the same UID and GID as on the host machine
RUN groupadd -g 1000 epalma && useradd -m -u 1000 -g 1000 epalma

# Install dependencies and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    python3 \
    python3-pip \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    libyaml-cpp-dev \
    vim \
    ssh \
    gdb \
    ffmpeg \
    clang-format \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install a compatible GCC version (11 for LIBTorch with CUDA 11.8)
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++-11 \
    gcc-11 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set GCC 11 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# Create third_party directory for external libraries
RUN mkdir -p /third_party

# Install cuDNN (ensure compatibility with CUDA 11.8)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8 libcudnn8-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install PyTorch C++ library for CUDA 11.8 (compatible with GCC 11)
RUN curl -L https://download.pytorch.org/libtorch/nightly/cu118/libtorch-cxx11-abi-shared-with-deps-latest.zip -o libtorch.zip && \
    unzip libtorch.zip -d third_party/ && \
    rm libtorch.zip

# Install Python packages
RUN pip3 install numpy matplotlib black

USER epalma

# Set default working directory
WORKDIR /workspace
