#include "storage/rollout.h"

namespace storage {

void RolloutStorage::initialize(const DictTensor& kl_params) {
  unsigned int num_steps_per_env = this->cfg_->runner_cfg.num_steps_per_env;
  unsigned int num_envs = this->cfg_->env_cfg.num_envs;

  unsigned int actor_obs_size = this->cfg_->actor_cfg.mlp_cfg.num_inputs;
  unsigned int critic_obs_size = this->cfg_->critic_cfg.mlp_cfg.num_inputs;
  unsigned int action_size = this->cfg_->actor_cfg.mlp_cfg.num_outputs;

  this->transitions_.actor_obs =
    torch::zeros({num_steps_per_env, num_envs, actor_obs_size}, this->device_);
  this->transitions_.critic_obs =
    torch::zeros({num_steps_per_env, num_envs, critic_obs_size}, this->device_);
  this->transitions_.actions =
    torch::zeros({num_steps_per_env, num_envs, action_size}, this->device_);
  this->transitions_.rewards = torch::zeros({num_steps_per_env, num_envs, 1}, this->device_);
  this->transitions_.advantages = torch::zeros({num_steps_per_env, num_envs, 1}, this->device_);
  this->transitions_.dones = torch::zeros({num_steps_per_env, num_envs, 1}, this->device_);
  this->transitions_.values = torch::zeros({num_steps_per_env, num_envs, 1}, this->device_);
  this->transitions_.log_probs = torch::zeros({num_steps_per_env, num_envs, 1}, this->device_);
  for (const auto& [key, value] : kl_params)
    this->transitions_.kl_params[key] =
      torch::zeros({num_steps_per_env, num_envs, value.size(0)}, this->device_);
  this->returns_ = torch::zeros({num_steps_per_env, num_envs, 1}, this->device_);
}

void RolloutStorage::push_back(const Transition& transition) {
  if (this->step_ == this->cfg_->runner_cfg.num_steps_per_env) {
    throw std::runtime_error("RolloutStorage is full");
  }
  this->transitions_.actor_obs.select(0, this->step_).copy_(transition.actor_obs);
  this->transitions_.critic_obs.select(0, this->step_).copy_(transition.critic_obs);
  this->transitions_.actions.select(0, this->step_).copy_(transition.actions);
  this->transitions_.rewards.select(0, this->step_).copy_(transition.rewards);
  this->transitions_.dones.select(0, this->step_).copy_(transition.dones);
  this->transitions_.values.select(0, this->step_).copy_(transition.values);
  this->transitions_.log_probs.select(0, this->step_).copy_(transition.log_probs);
  for (const auto& [key, value] : transition.kl_params)
    this->transitions_.kl_params[key].select(0, this->step_).copy_(value);

  this->step_++;
}

void RolloutStorage::compute_advantage(const Tensor& last_values, const float& gamma,
                                       const float& lambda) {
  Tensor advantage = torch::zeros({this->cfg_->env_cfg.num_envs, 1}, this->device_);
  Tensor next_value = last_values;
  for (int step = this->cfg_->runner_cfg.num_steps_per_env - 1; step >= 0; step--) {
    const Tensor next_is_not_terminal =
      1.0 - this->transitions_.dones.select(0, step).to(torch::kFloat);
    const Tensor delta = this->transitions_.rewards.select(0, step) +
                         gamma * next_value * next_is_not_terminal -
                         this->transitions_.values.select(0, step);
    advantage = delta + gamma * lambda * advantage * next_is_not_terminal;
    this->returns_.select(0, step).copy_(advantage + this->transitions_.values.select(0, step));
    next_value.copy_(this->transitions_.values.select(0, step));
  }
  this->transitions_.advantages.copy_(this->returns_ - this->transitions_.values);
  this->transitions_.advantages.copy_(
    (this->transitions_.advantages - this->transitions_.advantages.mean()) /
    (this->transitions_.advantages.std() + EPS));
}

void RolloutStorage::update_batches(std::vector<storage::Transition>& batches) {
  const unsigned& num_batches = this->cfg_->ppo_cfg.num_batches;
  const unsigned& num_epochs = this->cfg_->ppo_cfg.num_epochs;
  unsigned int batch_size =
    this->cfg_->env_cfg.num_envs * this->cfg_->runner_cfg.num_steps_per_env / num_batches;
  const Tensor random_indices = torch::randperm(batch_size * num_batches, this->device_);

  Transition flattened_transitions{
    .actor_obs = this->transitions_.actor_obs.view({-1, this->transitions_.actor_obs.size(2)}),
    .critic_obs = this->transitions_.critic_obs.view({-1, this->transitions_.critic_obs.size(2)}),
    .actions = this->transitions_.actions.view({-1, this->transitions_.actions.size(2)}),
    .rewards = this->returns_.view({-1, 1}),
    .advantages = this->transitions_.advantages.view({-1, 1}),
    .dones = this->transitions_.dones.view({-1, 1}),
    .values = this->transitions_.values.view({-1, 1}),
    .log_probs = this->transitions_.log_probs.view({-1, 1}),
  };
  for (const auto& [key, value] : this->transitions_.kl_params)
    flattened_transitions.kl_params[key] = value.view({-1, value.size(2)});

  batches.clear();
  for (unsigned int epoch = 0; epoch < num_epochs; ++epoch) {
    for (unsigned int i = 0; i < num_batches; ++i) {
      unsigned int start = i * batch_size;
      unsigned int end = (i + 1) * batch_size;

      const Tensor indices = random_indices.index({Slice(start, end)});

      Transition batch{.actor_obs = flattened_transitions.actor_obs.index({indices, Slice()}),
                       .critic_obs = flattened_transitions.critic_obs.index({indices, Slice()}),
                       .actions = flattened_transitions.actions.index({indices, Slice()}),
                       .rewards = flattened_transitions.rewards.index({indices, Slice()}),
                       .advantages = flattened_transitions.advantages.index({indices, Slice()}),
                       .dones = flattened_transitions.dones.index({indices, Slice()}),
                       .values = flattened_transitions.values.index({indices, Slice()}),
                       .log_probs = flattened_transitions.log_probs.index({indices, Slice()})};
      for (const auto& [key, value] : flattened_transitions.kl_params)
        batch.kl_params[key] = value.index({indices, Slice()});

      batches.push_back(batch);
    }
  }
}

}  // namespace storage