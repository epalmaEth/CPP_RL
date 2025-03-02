#pragma once

#include "configs/configs.h"
#include "env/env.h"
#include "utils/types.h"

namespace storage {

class ObservationBuffer {
 public:
  ObservationBuffer(const configs::CfgPointer& cfg, const unsigned int& num_actor_obs,
                    const unsigned int& num_critic_obs, const unsigned int& num_actions,
                    const Device& device)
    : cfg_(cfg),
      num_actor_obs_(num_actor_obs),
      num_critic_obs_(num_critic_obs),
      num_actions_(num_actions),
      device_(device),
      all_indices_(torch::ones({cfg->env_cfg.num_envs},
                               torch::TensorOptions().device(device).dtype(torch::kBool))) {
    this->initialize_();
  }

  void reset(env::Results& results, const Tensor& indices = {});
  void memorize(const env::Results& results, const Tensor& actions);
  const Tensor get_actor_obs() const {
    return this->buffer_actor_obs_.view({this->cfg_->env_cfg.num_envs, -1});
  }
  const Tensor get_critic_obs() const {
    return this->buffer_critic_obs_.view({this->cfg_->env_cfg.num_envs, -1});
  }
  unsigned int get_actor_obs_size() const {
    return this->buffer_actor_obs_.size(1) * this->buffer_actor_obs_.size(2);
  }
  unsigned int get_critic_obs_size() const {
    return this->buffer_critic_obs_.size(1) * this->buffer_critic_obs_.size(2);
  }

 private:
  void initialize_();
  const Tensor get_extended_obs_(const Tensor& obs, const Tensor& actions) const;
  const configs::CfgPointer cfg_;

  const Device device_;
  Tensor buffer_actor_obs_;
  Tensor buffer_critic_obs_;
  const Tensor all_indices_;
  const unsigned int num_actor_obs_;
  const unsigned int num_critic_obs_;
  const unsigned int num_actions_;
};

using ObservationBufferPointer = std::unique_ptr<ObservationBuffer>;
}  // namespace storage