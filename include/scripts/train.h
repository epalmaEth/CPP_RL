#pragma once

#include <torch/cuda.h>
#include <torch/torch.h>

#include <iostream>

#include "configs/configs.h"
#include "configs/load_yaml.h"
#include "env/env.h"
#include "runners/on_policy_runner.h"
#include "utils/task_manager.h"
#include "utils/types.h"

int train(const string &task) {
  utils::TaskManager task_manager;

  std::cout << "-------Train-------" << std::endl;
  const Device device =
      torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  std::cout << "Device: " << device << std::endl;

  std::cout << "-------Loading Cfg-------" << std::endl;
  configs::Cfg cfg = configs::load_config(task, "train");

  std::cout << "-------Creating Env-------" << std::endl;
  std::shared_ptr<env::Env> env = task_manager.create_env(
      task, cfg.runner_cfg.num_envs, cfg.runner_cfg.seed, device);
  env->initialize_states();
  auto [obs, info] = env->reset(env->get_all_indices());

  std::cout << "-------Creating Runner-------" << std::endl;
  runners::OnPolicyRunner runner(env, cfg, device);

  runner.learn();

  return 0;
}