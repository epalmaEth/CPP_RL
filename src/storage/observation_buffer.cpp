#include "storage/observation_buffer.h"

#include <torch/torch.h>

namespace storage {

void ObservationBuffer::reset(env::Results& results, const Tensor& indices) {
  const Tensor valid_indices = indices.defined() ? indices : this->all_indices_;
  const unsigned int num_resets = valid_indices.sum().item<int>();
  if (num_resets == 0) return;

  const Tensor actions =
    torch::zeros({num_resets, this->num_actions_}, torch::TensorOptions().device(this->device_));
  const Tensor extended_actor_obs =
    this->get_extended_obs_(results.actor_obs.index({valid_indices}), actions);
  const Tensor extended_critic_obs =
    this->get_extended_obs_(results.critic_obs.index({valid_indices}), actions);

  unsigned int memory_length = this->cfg_->runner_cfg.observation_memory_length;
  this->buffer_actor_obs_.index_put_(
    {valid_indices, Slice(), Slice()},
    extended_actor_obs.unsqueeze(1).expand({-1, memory_length, -1}));
  this->buffer_critic_obs_.index_put_(
    {valid_indices, Slice(), Slice()},
    extended_critic_obs.unsqueeze(1).expand({-1, memory_length, -1}));
}

void ObservationBuffer::memorize(const env::Results& results, const torch::Tensor& actions) {
  const Tensor extended_actor_obs = this->get_extended_obs_(results.actor_obs, actions);
  const Tensor extended_critic_obs = this->get_extended_obs_(results.critic_obs, actions);

  this->buffer_actor_obs_ = torch::roll(this->buffer_actor_obs_, -1, 1);
  this->buffer_critic_obs_ = torch::roll(this->buffer_critic_obs_, -1, 1);

  this->buffer_actor_obs_.select(1, -1).copy_(extended_actor_obs);
  this->buffer_critic_obs_.select(1, -1).copy_(extended_critic_obs);
}

void ObservationBuffer::initialize_() {
  unsigned int num_actor_obs = this->num_actor_obs_;
  unsigned int num_critic_obs = this->num_critic_obs_;
  if (this->cfg_->runner_cfg.observation_memory_store_action) {
    num_actor_obs += this->num_actions_;
    num_critic_obs += this->num_actions_;
  }
  unsigned int memory_length = this->cfg_->runner_cfg.observation_memory_length;

  this->buffer_actor_obs_ =
    torch::zeros({this->cfg_->env_cfg.num_envs, memory_length, num_actor_obs},
                 torch::TensorOptions().device(this->device_));
  this->buffer_critic_obs_ =
    torch::zeros({this->cfg_->env_cfg.num_envs, memory_length, num_critic_obs},
                 torch::TensorOptions().device(this->device_));
}

const Tensor ObservationBuffer::get_extended_obs_(const Tensor& obs, const Tensor& actions) const {
  if (this->cfg_->runner_cfg.observation_memory_store_action) return torch::cat({obs, actions}, 1);
  return obs;
}

}  // namespace storage