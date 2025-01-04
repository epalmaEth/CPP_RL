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

int play(const string& task) {
  torch::NoGradGuard no_grad;

  utils::TaskManager task_manager;

  std::cout << "-------Play-------" << std::endl;
  const Device device =
      torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  std::cout << "Device: " << device << std::endl;

  std::cout << "-------Loading Cfg-------" << std::endl;
  configs::Cfg cfg = configs::load_config(task, "play");

  std::cout << "-------Creating Env-------" << std::endl;
  std::shared_ptr<env::Env> env = task_manager.create_env(
      task, cfg.runner_cfg.num_envs, cfg.runner_cfg.seed, device);
  env->initialize_states();
  auto [obs, info] = env->reset(env->get_all_indices());

  std::cout << "-------Creating Runner-------" << std::endl;
  runners::OnPolicyRunner runner(env, cfg, device);

  std::cout << "-------Loading Model-------" << std::endl;
  runner.load_models("models_last.pt");

  std::cout << "-------Starting simulation-------" << std::endl;
  const auto& policy = runner.get_inference_policy();
  DictListTensor data;
  while (true) {
    auto step_result = env->step(policy(obs));
    env->update_render_data(data);

    if ((step_result.terminated | step_result.truncated).all().item<bool>()) {
      break;
    }

    obs.index_put_({Slice()}, step_result.obs);
  }
  std::cout << "-------Ending Simulation-------" << std::endl;
  env->close();

  std::cout << "-------Rendering-------" << std::endl;
  env->render(data);

  return 0;
}