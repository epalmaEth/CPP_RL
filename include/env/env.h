#pragma once

#include <torch/torch.h>

#include <string>

#include "utils/types.h"

namespace env {

class Env {
 public:
  explicit Env(const Device& device, int seed = -1, const int& num_envs = 1)
      : num_envs_(num_envs), device_(device) {
    if (seed < 0) seed = this->sample_random_seed_();
    torch::manual_seed(seed);
    this->all_indices_ =
        torch::arange(0, num_envs).to(torch::kInt32).to(device);
  }

  virtual ~Env() = default;

  virtual void initialize_states() = 0;
  virtual std::pair<Tensor, DictTensor> reset(const Tensor& indices) = 0;
  virtual void close() = 0;
  virtual int get_action_size() const = 0;
  virtual Tensor sample_action() const = 0;
  virtual void update_render_data(DictListTensor& data) const = 0;
  virtual void render(const DictListTensor& data) const = 0;

  std::pair<Tensor, DictTensor> reset(int seed, const Tensor& indices) {
    torch::manual_seed(seed);
    return this->reset(indices);
  };

  StepResult step(const Tensor& action) {
    this->update_step_state_(action);
    this->iteration_ += 1;
    return this->step_result_;
  };

  int get_num_envs() const { return this->num_envs_; }
  Device get_device() const { return this->device_; }
  Tensor get_state() const { return this->state_; }
  Tensor get_observation() const { return this->step_result_.observation; }
  Tensor get_reward() const { return this->step_result_.reward; }
  Tensor get_terminated() const { return this->step_result_.terminated; }
  Tensor get_truncated() const { return this->step_result_.truncated; }
  DictTensor get_info() const { return this->step_result_.info; }
  Tensor get_all_indices() const { return this->all_indices_; }

 protected:
  virtual void sample_state_(const Tensor& indices) = 0;
  virtual void update_state_(const Tensor& action) = 0;
  virtual void compute_observations_() = 0;
  virtual void compute_reward_() = 0;
  virtual void compute_terminated_() = 0;
  virtual void compute_truncated_() = 0;
  virtual void compute_info_() = 0;

  int sample_random_seed_() const {
    return torch::randint(0, std::numeric_limits<int>::max(), {1}).item<int>();
  }

  void update_step_state_(const Tensor& action) {
    this->update_state_(action);
    this->compute_observations_();
    this->compute_reward_();
    this->compute_terminated_();
    this->compute_truncated_();
    this->compute_info_();
  }

  int num_envs_;
  int max_iterations_;
  Device device_;

  Tensor iteration_;
  StepResult step_result_;
  Tensor state_;
  Tensor all_indices_;
};
}  // namespace env