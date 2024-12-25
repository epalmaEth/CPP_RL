# C++ PPO Implementation

This repository implements Proximal Policy Optimization (PPO) in C++ for an environment, inspired by PyTorch's structure.

## Features
- Modular design for PPO and environment
- Dockerized build environment for easy setup

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cpp-ppo.git
   cd cpp-ppo
   ```
2. Build container
    ```bash
    docker compose -f docker/docker-compose.yaml build
    ```
3. Run container
    ```bashZ
    docker compose -f docker/docker-compose.yaml run --rm cpp_rl
    ```
Or use `Ctrl+Shift+B` to build and run the container