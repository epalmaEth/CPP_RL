#pragma once

#include <iostream>

#include "utils/types.h"
#include "utils/utils.h"

namespace configs {

struct NormalizerCfg {
  unsigned int num_inputs = 0;
  const string type;

  explicit NormalizerCfg(const string& type) : type(type) {}

  void update(const unsigned int& num_obs) { this->num_inputs = num_obs; }

  friend std::ostream& operator<<(std::ostream& os, const NormalizerCfg& cfg) {
    os << "         num_inputs: " << cfg.num_inputs << " (modified depending on env)" << std::endl;
    os << "         type: " << cfg.type;
    return os;
  }
};

struct MLPCfg {
  unsigned int num_inputs = 0;
  unsigned int num_outputs = 0;
  unsigned int width;
  const unsigned int depth;
  const string activation;

  MLPCfg(const unsigned int& width, const unsigned int& depth, const string& activation)
    : width(width), depth(depth), activation(activation) {}

  void update(const unsigned int& num_obs, const unsigned int& action_size) {
    this->num_inputs = num_obs;
    this->num_outputs = action_size;
    this->width = utils::next_power_of_2(num_obs, this->width);
  }

  friend std::ostream& operator<<(std::ostream& os, const MLPCfg& cfg) {
    os << "         num_inputs: " << cfg.num_inputs << " (modified depending on env)" << std::endl;
    os << "         num_outputs: " << cfg.num_outputs << " (modified depending on env)"
       << std::endl;
    os << "         width: " << cfg.width << " (modified depending on env)" << std::endl;
    os << "         depth: " << cfg.depth << std::endl;
    os << "         activation: " << cfg.activation;
    return os;
  }
};

struct DistributionCfg {
  unsigned int num_inputs = 0;
  const float init_noise_std;
  const string type;
  Tensor action_min;
  Tensor action_max;

  DistributionCfg(const float& init_noise_std, const string& type)
    : init_noise_std(init_noise_std), type(type) {}

  void update(const Tensor& action_min, const Tensor& action_max) {
    this->num_inputs = action_min.size(0);
    this->action_min = action_min.clone();
    this->action_max = action_max.clone();
  }

  friend std::ostream& operator<<(std::ostream& os, const DistributionCfg& cfg) {
    os << "        num_inputs: " << cfg.num_inputs << " (modified depending on env)" << std::endl;
    os << "        init_noise_std: " << cfg.init_noise_std << std::endl;
    os << "        type: " << cfg.type << std::endl;
    os << "        action_min: ";
    for (int64_t i = 0; i < cfg.num_inputs; ++i) {
      os << cfg.action_min[i].item<float>() << " ";
    }
    os << "(modified depending on env)" << std::endl;

    // Print action_max
    os << "        action_max: ";
    for (int64_t i = 0; i < cfg.num_inputs; ++i) {
      os << cfg.action_max[i].item<float>() << " ";
    }
    os << "(modified depending on env)" << std::endl;
    return os;
  }
};

struct ActorCfg {
  NormalizerCfg normalizer_cfg;
  MLPCfg mlp_cfg;
  DistributionCfg distribution_cfg;

  ActorCfg(const NormalizerCfg& normalizer_cfg, const MLPCfg& mlp_cfg,
           const DistributionCfg& distribution_cfg)
    : normalizer_cfg(normalizer_cfg), mlp_cfg(mlp_cfg), distribution_cfg(distribution_cfg) {}

  void update(const unsigned int& num_actor_obs, const Tensor& action_min,
              const Tensor& action_max) {
    this->normalizer_cfg.update(num_actor_obs);
    this->mlp_cfg.update(num_actor_obs, action_min.size(0));
    this->distribution_cfg.update(action_min, action_max);
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

  CriticCfg(const NormalizerCfg& normalizer_cfg, const MLPCfg& mlp_cfg)
    : normalizer_cfg(normalizer_cfg), mlp_cfg(mlp_cfg) {}

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
  const float value_loss_coef;
  const float clip_param;
  const bool use_clipped_value_loss;
  // -- Surrogate loss
  const float desired_kl;
  const float entropy_coef;
  const float gamma;
  const float lam;
  const float max_grad_norm;
  // -- Training
  const float learning_rate;
  const float min_learning_rate;
  const float max_learning_rate;
  const unsigned int num_epochs;
  const unsigned int num_batches;
  const string learning_rate_schedule;

  PPOCfg(const float& value_loss_coef, const float& clip_param, const bool& use_clipped_value_loss,
         const float& desired_kl, const float& entropy_coef, const float& gamma, const float& lam,
         const float& max_grad_norm, const float& learning_rate, const float& min_learning_rate,
         const float& max_learning_rate, const unsigned int& num_epochs,
         const unsigned int& num_batches, const string& learning_rate_schedule)
    : value_loss_coef(value_loss_coef),
      clip_param(clip_param),
      use_clipped_value_loss(use_clipped_value_loss),
      desired_kl(desired_kl),
      entropy_coef(entropy_coef),
      gamma(gamma),
      lam(lam),
      max_grad_norm(max_grad_norm),
      learning_rate(learning_rate),
      min_learning_rate(min_learning_rate),
      max_learning_rate(max_learning_rate),
      num_epochs(num_epochs),
      num_batches(num_batches),
      learning_rate_schedule(learning_rate_schedule) {}

  friend std::ostream& operator<<(std::ostream& os, const PPOCfg& cfg) {
    os << "    value_loss_coef: " << cfg.value_loss_coef << std::endl;
    os << "    clip_param: " << cfg.clip_param << std::endl;
    os << "    use_clipped_value_loss: " << cfg.use_clipped_value_loss << std::endl;
    os << "    desired_kl: " << cfg.desired_kl << std::endl;
    os << "    entropy_coef: " << cfg.entropy_coef << std::endl;
    os << "    gamma: " << cfg.gamma << std::endl;
    os << "    lam: " << cfg.lam << std::endl;
    os << "    max_grad_norm: " << cfg.max_grad_norm << std::endl;
    os << "    learning_rate: " << cfg.learning_rate << std::endl;
    os << "    min_learning_rate: " << cfg.min_learning_rate << std::endl;
    os << "    max_learning_rate: " << cfg.max_learning_rate << std::endl;
    os << "    num_epochs: " << cfg.num_epochs << std::endl;
    os << "    num_batches: " << cfg.num_batches << std::endl;
    os << "    learning_rate_schedule: " << cfg.learning_rate_schedule;
    return os;
  }
};

struct RunnerCfg {
  // -- Learning
  const unsigned int max_iterations;
  const unsigned int num_steps_per_env;
  const unsigned int observation_memory_length;
  const bool observation_memory_store_action;
  // -- Saving
  const unsigned int save_interval;
  // -- Logging
  const unsigned int logging_buffer;
  const unsigned int logging_warmup;

  RunnerCfg(const unsigned int& max_iterations, const unsigned int& num_steps_per_env,
            const unsigned int& observation_memory_length,
            const bool& observation_memory_store_action, const unsigned int& save_interval,
            const unsigned int& logging_buffer, const unsigned int& logging_warmup)
    : max_iterations(max_iterations),
      num_steps_per_env(num_steps_per_env),
      observation_memory_length(observation_memory_length),
      observation_memory_store_action(observation_memory_store_action),
      save_interval(save_interval),
      logging_buffer(logging_buffer),
      logging_warmup(logging_warmup) {}

  friend std::ostream& operator<<(std::ostream& os, const RunnerCfg& cfg) {
    os << "    max_iterations: " << cfg.max_iterations << std::endl;
    os << "    num_steps_per_env: " << cfg.num_steps_per_env << std::endl;
    os << "    observation_memory_length: " << cfg.observation_memory_length << std::endl;
    os << "    observation_memory_store_action: "
       << (cfg.observation_memory_store_action ? "true" : "false") << std::endl;
    os << "    save_interval: " << cfg.save_interval << std::endl;
    os << "    logging_buffer: " << cfg.logging_buffer << std::endl;
    os << "    logging_warmup: " << cfg.logging_warmup;
    return os;
  }
};

struct EnvCfg {
  const int run_id;
  const string task;
  const unsigned int num_envs;
  const int seed;
  const int max_iterations;
  const string integrator;
  const float dt;

  EnvCfg(const int& run_id, const string& task, const unsigned int& num_envs, const int& seed,
         int max_iterations, const string& integrator, const float& dt)
    : run_id(run_id),
      task(task),
      num_envs(num_envs),
      seed(seed),
      max_iterations(max_iterations),
      integrator(integrator),
      dt(dt) {}

  friend std::ostream& operator<<(std::ostream& os, const EnvCfg& cfg) {
    os << "    run_id: " << cfg.run_id << std::endl;
    os << "    task: " << cfg.task << std::endl;
    os << "    num_envs: " << cfg.num_envs << std::endl;
    os << "    seed: " << cfg.seed << std::endl;
    os << "    max_iterations: " << cfg.max_iterations << std::endl;
    os << "    integrator: " << cfg.integrator << std::endl;
    os << "    dt: " << cfg.dt;
    return os;
  }
};

struct Cfg {
  const EnvCfg env_cfg;
  const RunnerCfg runner_cfg;
  const PPOCfg ppo_cfg;
  ActorCfg actor_cfg;
  CriticCfg critic_cfg;

  Cfg(const EnvCfg& env_cfg, const RunnerCfg& runner_cfg, const PPOCfg& ppo_cfg,
      const ActorCfg& actor_cfg, const CriticCfg& critic_cfg)
    : env_cfg(env_cfg),
      runner_cfg(runner_cfg),
      ppo_cfg(ppo_cfg),
      actor_cfg(actor_cfg),
      critic_cfg(critic_cfg) {}

  void update(const unsigned int& num_actor_obs, const unsigned int& num_critic_obs,
              const Tensor& action_min, const Tensor& action_max) {
    this->actor_cfg.update(num_actor_obs, action_min, action_max);
    this->critic_cfg.update(num_critic_obs);
  }

  friend std::ostream& operator<<(std::ostream& os, const Cfg& cfg) {
    os << "env: \n" << cfg.env_cfg << std::endl;
    os << "runner: \n" << cfg.runner_cfg << std::endl;
    os << "ppo: \n" << cfg.ppo_cfg << std::endl;
    os << "actor: \n" << cfg.actor_cfg << std::endl;
    os << "critic: \n" << cfg.critic_cfg << std::endl;
    return os;
  }
};

using CfgPointer = std::shared_ptr<Cfg>;

}  // namespace configs