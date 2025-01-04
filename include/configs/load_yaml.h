#pragma once

#include "configs/configs.h"
#include "utils/types.h"
#include "yaml-cpp/yaml.h"

namespace configs {

inline Cfg load_config(const string& task, const string& task_mode) {
  string config_path = "data/yaml/" + task + "/config.yaml";

  YAML::Node config = YAML::LoadFile(config_path);

  Cfg cfg;

  const auto& runner_cfg = config["runner"];
  cfg.runner_cfg.task = task;
  cfg.runner_cfg.num_envs =
      task_mode == "train" ? runner_cfg["num_envs"].as<unsigned int>() : 1;
  cfg.runner_cfg.seed = runner_cfg["seed"].as<int>();
  cfg.runner_cfg.max_iterations =
      runner_cfg["max_iterations"].as<unsigned int>();
  cfg.runner_cfg.num_steps_per_env =
      runner_cfg["num_steps_per_env"].as<unsigned int>();
  cfg.runner_cfg.save_interval = runner_cfg["save_interval"].as<int>();
  cfg.runner_cfg.save_dir = runner_cfg["save_dir"].as<string>() + task + "/";
  cfg.runner_cfg.load_dir = runner_cfg["load_dir"].as<string>() + task + "/";
  cfg.runner_cfg.logging_buffer =
      runner_cfg["logging_buffer"].as<unsigned int>();

  // PPO Configuration
  const auto& ppo_cfg = config["ppo"];
  cfg.ppo_cfg.value_loss_coef = ppo_cfg["value_loss_coef"].as<float>();
  cfg.ppo_cfg.clip_param = ppo_cfg["clip_param"].as<float>();
  cfg.ppo_cfg.use_clipped_value_loss =
      ppo_cfg["use_clipped_value_loss"].as<bool>();
  cfg.ppo_cfg.desired_kl = ppo_cfg["desired_kl"].as<float>();
  cfg.ppo_cfg.entropy_coef = ppo_cfg["entropy_coef"].as<float>();
  cfg.ppo_cfg.gamma = ppo_cfg["gamma"].as<float>();
  cfg.ppo_cfg.lam = ppo_cfg["lam"].as<float>();
  cfg.ppo_cfg.max_grad_norm = ppo_cfg["max_grad_norm"].as<float>();
  cfg.ppo_cfg.learning_rate = ppo_cfg["learning_rate"].as<float>();
  cfg.ppo_cfg.num_epochs = ppo_cfg["num_epochs"].as<unsigned int>();
  cfg.ppo_cfg.num_batches = ppo_cfg["num_batches"].as<unsigned int>();
  cfg.ppo_cfg.learning_rate_schedule =
      ppo_cfg["learning_rate_schedule"].as<string>();

  // Actor-Critic Configuration (same as before)
  const auto& actor_cfg = config["actor"];
  cfg.actor_cfg.normalizer_cfg.type =
      actor_cfg["normalizer"]["type"].as<string>();
  cfg.actor_cfg.mlp_cfg.width = actor_cfg["mlp"]["width"].as<unsigned int>();
  cfg.actor_cfg.mlp_cfg.depth = actor_cfg["mlp"]["depth"].as<unsigned int>();
  cfg.actor_cfg.mlp_cfg.activation =
      actor_cfg["mlp"]["activation"].as<string>();
  cfg.actor_cfg.distribution_cfg.init_noise_std =
      actor_cfg["distribution"]["init_noise_std"].as<float>();
  cfg.actor_cfg.distribution_cfg.type =
      actor_cfg["distribution"]["type"].as<string>();

  const auto& critic_cfg = config["critic"];
  cfg.critic_cfg.normalizer_cfg.type =
      critic_cfg["normalizer"]["type"].as<string>();
  cfg.critic_cfg.mlp_cfg.width = critic_cfg["mlp"]["width"].as<unsigned int>();
  cfg.critic_cfg.mlp_cfg.depth = critic_cfg["mlp"]["depth"].as<unsigned int>();
  cfg.critic_cfg.mlp_cfg.activation =
      critic_cfg["mlp"]["activation"].as<string>();

  return cfg;
}

}  // namespace configs