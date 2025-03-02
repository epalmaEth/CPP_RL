#pragma once

#include "configs/configs.h"
#include "env.h"
#include "physics_based_envs/pendulum.h"
#include "physics_based_envs/pendulum_cart.h"

namespace env {

class TaskManager {
 public:
  static EnvPointer create(const string& task, const configs::EnvCfg& cfg, const Device& device) {
    if (task == "pendulum") return std::make_unique<PendulumEnv>(cfg, device);
    if (task == "pendulum_cart") return std::make_unique<PendulumCartEnv>(cfg, device);
    throw std::invalid_argument("Unknown task: " + task);
  }
};

}  // namespace env