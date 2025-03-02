#pragma once

#include <yaml-cpp/yaml.h>

#include "configs.h"
#include "utils/types.h"
#include "utils/utils.h"

namespace configs {

inline const CfgPointer load_config(const string& task, const bool& play) {
  const string task_path = "data/" + task;
  int last_train_run_id = utils::last_run_id(task_path);
  const string last_train_run_path = task_path + "/run_" + std::to_string(last_train_run_id);

  const string train_config_path = play ? last_train_run_path + "/config.yaml" : "yaml/train.yaml";
  const string play_config_path = "yaml/play.yaml";

  YAML::Node train_config = YAML::LoadFile(train_config_path);
  YAML::Node play_config = YAML::LoadFile(play_config_path);

  int play_run_id = play_config["run_id"].as<int>();

  play_run_id = play_run_id < 0 ? last_train_run_id : play_run_id;

  // Env Configuration
  const auto& env_yaml = train_config["env"];
  const EnvCfg env_cfg{play ? play_run_id : last_train_run_id,
                       task,
                       play ? 1 : env_yaml["num_envs"].as<unsigned int>(),
                       play ? -1 : env_yaml["seed"].as<int>(),
                       env_yaml["max_iterations"].as<int>(),
                       env_yaml["integrator"] ? env_yaml["integrator"].as<std::string>() : "",
                       env_yaml["dt"] ? env_yaml["dt"].as<float>() : POS_INF_F};

  // Runner Configuration
  const auto& runner_yaml = train_config["runner"];
  const RunnerCfg runner_cfg{runner_yaml["max_iterations"].as<unsigned int>(),
                             runner_yaml["num_steps_per_env"].as<unsigned int>(),
                             runner_yaml["observation_memory_length"].as<unsigned int>(),
                             runner_yaml["observation_memory_store_action"].as<bool>(),
                             runner_yaml["save_interval"].as<unsigned int>(),
                             runner_yaml["logging_buffer"].as<unsigned int>(),
                             runner_yaml["logging_warmup"].as<unsigned int>()};

  // PPO Configuration
  const auto& ppo_yaml = train_config["ppo"];
  const PPOCfg ppo_cfg{ppo_yaml["value_loss_coef"].as<float>(),
                       ppo_yaml["clip_param"].as<float>(),
                       ppo_yaml["use_clipped_value_loss"].as<bool>(),
                       ppo_yaml["desired_kl"].as<float>(),
                       ppo_yaml["entropy_coef"].as<float>(),
                       ppo_yaml["gamma"].as<float>(),
                       ppo_yaml["lam"].as<float>(),
                       ppo_yaml["max_grad_norm"].as<float>(),
                       ppo_yaml["learning_rate"].as<float>(),
                       ppo_yaml["min_learning_rate"].as<float>(),
                       ppo_yaml["max_learning_rate"].as<float>(),
                       ppo_yaml["num_epochs"].as<unsigned int>(),
                       ppo_yaml["num_batches"].as<unsigned int>(),
                       ppo_yaml["learning_rate_schedule"].as<string>()};

  // Actor-Critic Configuration
  const auto& actor_normalizer_yaml = train_config["actor"]["normalizer"];
  const NormalizerCfg actor_normalizer_cfg{actor_normalizer_yaml["type"].as<string>()};
  const auto& actor_mlp_yaml = train_config["actor"]["mlp"];
  const MLPCfg actor_mlp_cfg{actor_mlp_yaml["width"].as<unsigned int>(),
                             actor_mlp_yaml["depth"].as<unsigned int>(),
                             actor_mlp_yaml["activation"].as<string>()};
  const auto& actor_distribution_yaml = train_config["actor"]["distribution"];
  const DistributionCfg actor_distribution_cfg{
    actor_distribution_yaml["init_noise_std"].as<float>(),
    actor_distribution_yaml["type"].as<string>()};
  const ActorCfg actor_cfg{actor_normalizer_cfg, actor_mlp_cfg, actor_distribution_cfg};

  const auto& critic_normalizer_yaml = train_config["critic"]["normalizer"];
  const NormalizerCfg critic_normalizer_cfg{critic_normalizer_yaml["type"].as<string>()};
  const auto& critic_mlp_yaml = train_config["critic"]["mlp"];
  const MLPCfg critic_mlp_cfg{critic_mlp_yaml["width"].as<unsigned int>(),
                              critic_mlp_yaml["depth"].as<unsigned int>(),
                              critic_mlp_yaml["activation"].as<string>()};
  const CriticCfg critic_cfg{critic_normalizer_cfg, critic_mlp_cfg};

  return std::make_shared<Cfg>(env_cfg, runner_cfg, ppo_cfg, actor_cfg, critic_cfg);
}

}  // namespace configs