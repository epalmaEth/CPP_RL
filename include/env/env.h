#pragma once

#include <torch/torch.h>

#include "configs/configs.h"
#include "utils/types.h"

namespace env {

struct Results {
  Tensor actor_obs;
  Tensor critic_obs;
  Tensor rewards;
  Tensor terminated;
  Tensor truncated;
  DictTensor info;
};

class Env {
 public:
  Env(const configs::EnvCfg& cfg, const Device& device)
    : cfg_(cfg),
      device_(device),
      max_iterations_(cfg.max_iterations),
      all_indices_(
        torch::ones({cfg.num_envs}, torch::TensorOptions().device(device).dtype(torch::kBool))) {
    torch::manual_seed(cfg.seed < 0 ? this->sample_random_seed_() : cfg.seed);
  }

  virtual ~Env() = default;

  virtual void initialize() {
    this->state_ = torch::zeros({this->cfg_.num_envs, this->get_state_size_()}, this->device_);
    this->iteration_ = torch::zeros({this->cfg_.num_envs}, this->device_);
    this->initialize_render_();
  }
  virtual void close() = 0;
  virtual unsigned int get_actor_obs_size() const = 0;
  virtual unsigned int get_critic_obs_size() const { return this->get_actor_obs_size(); }
  virtual unsigned int get_action_size() const = 0;
  virtual const Tensor get_action_min() const {
    return torch::full((this->get_action_size()), NEG_INF_F, this->device_);
  }
  virtual const Tensor get_action_max() const {
    return torch::full((this->get_action_size()), POS_INF_F, this->device_);
  }
  virtual const Tensor sample_action() const = 0;
  virtual void update_render_trajectory(const Results& results) const = 0;
  virtual void render() const = 0;

  void reset(Results& results, const Tensor& indices = {}) {
    const Tensor& valid_indices = indices.defined() ? indices : this->all_indices_;
    const unsigned int num_resets = valid_indices.sum().item<int>();
    if (num_resets == 0) return;
    this->reset_state_(num_resets, valid_indices);
    this->iteration_.index_put_({valid_indices}, 0);
    this->update_actor_obs_(results);
    this->update_critic_obs_(results);
    this->update_info_(results);
  }

  void step(Results& results, const Tensor& action) {
    this->iteration_ += 1;
    this->update_state_(action);
    this->update_results_(results);
  }

 protected:
  virtual unsigned int get_state_size_() const = 0;
  virtual void reset_state_(const int& num_resets, const Tensor& indices) = 0;
  virtual void update_state_(const Tensor& action) = 0;

  int sample_random_seed_() const {
    return torch::randint(0, std::numeric_limits<int>::max(), {1}).item<int>();
  }

  virtual void update_actor_obs_(Results& results) = 0;
  virtual void update_critic_obs_(Results& results) = 0;
  virtual void update_rewards_(Results& results) = 0;
  virtual void update_terminated_(Results& results) { return; }
  virtual void update_truncated_(Results& results) {
    results.truncated.copy_(this->iteration_ >= this->max_iterations_);
  }
  virtual void update_info_(Results& results) { return; }
  void update_results_(Results& results) {
    this->update_actor_obs_(results);
    this->update_critic_obs_(results);
    this->update_terminated_(results);
    this->update_truncated_(results);
    this->update_rewards_(results);
    this->update_info_(results);
  }
  virtual void initialize_render_() = 0;
  virtual string task_name_() const = 0;

  const configs::EnvCfg cfg_;
  const Device device_;
  Tensor iteration_;
  Tensor state_;
  const Tensor all_indices_;
  unsigned int max_iterations_;
};

using EnvPointer = std::unique_ptr<Env>;
}  // namespace env