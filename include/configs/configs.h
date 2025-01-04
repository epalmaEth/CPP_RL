#pragma once

#include <iostream>

#include "utils/types.h"
#include "utils/utils.h"

namespace configs {

struct NormalizerCfg {
  unsigned int num_inputs = 0;
  string type;

  void update(const unsigned int& num_obs) { this->num_inputs = num_obs; }

  friend std::ostream& operator<<(std::ostream& os, const NormalizerCfg& cfg) {
    os << "         num_inputs: " << cfg.num_inputs
       << " (modified depending on env)" << std::endl;
    os << "         type: " << cfg.type;
    return os;
  }
};

struct MLPCfg {
  unsigned int num_inputs = 0;
  unsigned int num_outputs = 0;
  unsigned int width;
  unsigned int depth;
  string activation;

  void update(const unsigned int& num_obs, const unsigned int& action_size) {
    this->num_inputs = num_obs;
    this->num_outputs = action_size;
    this->width = utils::next_power_of_2(num_obs, this->width);
  }

  friend std::ostream& operator<<(std::ostream& os, const MLPCfg& cfg) {
    os << "         num_inputs: " << cfg.num_inputs
       << " (modified depending on env)" << std::endl;
    os << "         num_outputs: " << cfg.num_outputs
       << " (modified depending on env)" << std::endl;
    os << "         width: " << cfg.width << " (modified depending on env)"
       << std::endl;
    os << "         depth: " << cfg.depth << std::endl;
    os << "         activation: " << cfg.activation;
    return os;
  }
};

struct DistributionCfg {
  unsigned int num_inputs = 0;
  float init_noise_std;
  string type;

  void update(const unsigned int& action_size) {
    this->num_inputs = action_size;
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const DistributionCfg& cfg) {
    os << "        num_inputs: " << cfg.num_inputs
       << " (modified depending on env)" << std::endl;
    os << "        init_noise_std: " << cfg.init_noise_std << std::endl;
    os << "        type: " << cfg.type;
    return os;
  }
};

struct ActorCfg {
  NormalizerCfg normalizer_cfg;
  MLPCfg mlp_cfg;
  DistributionCfg distribution_cfg;

  void update(const unsigned int& num_actor_obs,
              const unsigned int& action_size) {
    this->normalizer_cfg.update(num_actor_obs);
    this->mlp_cfg.update(num_actor_obs, action_size);
    this->distribution_cfg.update(action_size);
  }

  friend std::ostream& operator<<(std::ostream& os, const ActorCfg& cfg) {
    os << "    normalizer: \n" << cfg.normalizer_cfg << std::endl;
    os << "    mlp: \n" << cfg.mlp_cfg << std::endl;
    os << "    distribution: \n" << cfg.distribution_cfg;
    return os;
  }
};

struct CriticCfg {
  NormalizerCfg normalizer_cfg;
  MLPCfg mlp_cfg;

  void update(const unsigned int& num_critic_obs) {
    this->normalizer_cfg.update(num_critic_obs);
    this->mlp_cfg.update(num_critic_obs, 1);
  }

  friend std::ostream& operator<<(std::ostream& os, const CriticCfg& cfg) {
    os << "    normalizer: \n" << cfg.normalizer_cfg << std::endl;
    os << "    mlp: \n" << cfg.mlp_cfg;
    return os;
  }
};

struct PPOCfg {
  // -- Value loss
  float value_loss_coef;
  float clip_param;
  bool use_clipped_value_loss;
  // -- Surrogate loss
  float desired_kl;
  float entropy_coef;
  float gamma;
  float lam;
  float max_grad_norm;
  // -- Training
  float learning_rate;
  unsigned int num_epochs;
  unsigned int num_batches;
  string learning_rate_schedule;

  friend std::ostream& operator<<(std::ostream& os, const PPOCfg& cfg) {
    os << "    value_loss_coef: " << cfg.value_loss_coef << std::endl;
    os << "    clip_param: " << cfg.clip_param << std::endl;
    os << "    use_clipped_value_loss: " << cfg.use_clipped_value_loss
       << std::endl;
    os << "    desired_kl: " << cfg.desired_kl << std::endl;
    os << "    entropy_coef: " << cfg.entropy_coef << std::endl;
    os << "    gamma: " << cfg.gamma << std::endl;
    os << "    lam: " << cfg.lam << std::endl;
    os << "    max_grad_norm: " << cfg.max_grad_norm << std::endl;
    os << "    learning_rate: " << cfg.learning_rate << std::endl;
    os << "    num_epochs: " << cfg.num_epochs << std::endl;
    os << "    num_batches: " << cfg.num_batches << std::endl;
    os << "    learning_rate_schedule: " << cfg.learning_rate_schedule;
    return os;
  }
};

struct RunnerCfg {
  // -- Env
  string task;
  unsigned int num_envs;
  int seed;
  // -- Learning
  unsigned int max_iterations;
  unsigned int num_steps_per_env;
  // -- Saving
  int save_interval;
  string save_dir;
  // --Loading
  string load_dir;
  // -- Logging
  unsigned int logging_buffer;

  friend std::ostream& operator<<(std::ostream& os, const RunnerCfg& cfg) {
    os << "    task: " << cfg.task << std::endl;
    os << "    num_envs: " << cfg.num_envs << std::endl;
    os << "    seed: " << cfg.seed << std::endl;
    os << "    max_iterations: " << cfg.max_iterations << std::endl;
    os << "    num_steps_per_env: " << cfg.num_steps_per_env << std::endl;
    os << "    save_interval: " << cfg.save_interval << std::endl;
    os << "    save_dir: " << cfg.save_dir << std::endl;
    os << "    load_dir: " << cfg.load_dir << std::endl;
    os << "    logging_buffer: " << cfg.logging_buffer;
    return os;
  }
};

struct Cfg {
  RunnerCfg runner_cfg;
  PPOCfg ppo_cfg;
  ActorCfg actor_cfg;
  CriticCfg critic_cfg;

  void update(const unsigned int& num_actor_obs,
              const unsigned int& num_critic_obs,
              const unsigned int& action_size) {
    this->actor_cfg.update(num_actor_obs, action_size);
    this->critic_cfg.update(num_critic_obs);
  }

  friend std::ostream& operator<<(std::ostream& os, const Cfg& cfg) {
    os << "runner: \n" << cfg.runner_cfg << std::endl;
    os << "ppo: \n" << cfg.ppo_cfg << std::endl;
    os << "actor: \n" << cfg.actor_cfg << std::endl;
    os << "critic: \n" << cfg.critic_cfg << std::endl;
    return os;
  }
};

}  // namespace configs