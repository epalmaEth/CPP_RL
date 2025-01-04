#include "storage/rollout.h"

namespace storage {

void RolloutStorage::initialize(const DictTensor &kl_params) {
  DictTensor all_kl_params;
  const int &num_steps_per_env = this->cfg_.runner_cfg.num_steps_per_env;
  const int &num_envs = this->cfg_.runner_cfg.num_envs;

  for (const auto &kl_param : kl_params)
    all_kl_params[kl_param.first] = torch::zeros(
        {num_steps_per_env, num_envs, kl_param.second.size(0)}, this->device_);

  const int &actor_obs_size = this->cfg_.actor_cfg.mlp_cfg.num_inputs;
  const int &critic_obs_size = this->cfg_.critic_cfg.mlp_cfg.num_inputs;
  const int &action_size = this->cfg_.actor_cfg.mlp_cfg.num_outputs;

  this->transitions_.actor_obs = torch::zeros(
      {num_steps_per_env, num_envs, actor_obs_size}, this->device_);
  this->transitions_.critic_obs = torch::zeros(
      {num_steps_per_env, num_envs, critic_obs_size}, this->device_);
  this->transitions_.actions =
      torch::zeros({num_steps_per_env, num_envs, action_size}, this->device_);
  this->transitions_.rewards =
      torch::zeros({num_steps_per_env, num_envs, 1}, this->device_);
  this->transitions_.advantages =
      torch::zeros({num_steps_per_env, num_envs, 1}, this->device_);
  this->transitions_.dones =
      torch::zeros({num_steps_per_env, num_envs, 1}, this->device_);
  this->transitions_.values =
      torch::zeros({num_steps_per_env, num_envs, 1}, this->device_);
  this->transitions_.log_probs =
      torch::zeros({num_steps_per_env, num_envs, 1}, this->device_);
  this->transitions_.kl_params = all_kl_params;
}

void RolloutStorage::insert(const Transition &transition) {
  if (this->step_ == this->cfg_.runner_cfg.num_steps_per_env) {
    throw std::runtime_error("RolloutStorage is full");
  }
  this->transitions_.actor_obs.index_put_({this->step_, Slice(), Slice()},
                                          transition.actor_obs.clone());
  this->transitions_.critic_obs.index_put_({this->step_, Slice(), Slice()},
                                           transition.critic_obs.clone());
  this->transitions_.actions.index_put_({this->step_, Slice(), Slice()},
                                        transition.actions.clone());
  this->transitions_.rewards.index_put_({this->step_, Slice(), Slice()},
                                        transition.rewards.clone());
  this->transitions_.dones.index_put_({this->step_, Slice(), Slice()},
                                      transition.dones.clone());
  this->transitions_.values.index_put_({this->step_, Slice(), Slice()},
                                       transition.values.clone());
  this->transitions_.log_probs.index_put_({this->step_, Slice(), Slice()},
                                          transition.log_probs.clone());
  for (const auto &kl_param : transition.kl_params)
    this->transitions_.kl_params[kl_param.first].index_put_(
        {this->step_, Slice(), Slice()}, kl_param.second.clone());
  this->step_++;
}

void RolloutStorage::compute_cumulative_rewards(const Tensor &last_values,
                                                const float &gamma,
                                                const float &lambda) {
  Tensor advantage =
      torch::zeros({this->cfg_.runner_cfg.num_envs, 1}, this->device_);
  Tensor next_value = last_values;
  for (int step = this->cfg_.runner_cfg.num_steps_per_env - 1; step >= 0;
       step--) {
    const Tensor next_is_not_terminal =
        1.0 - this->transitions_.dones.index({step, Slice(), Slice()})
                  .to(torch::kFloat64);
    const Tensor delta =
        this->transitions_.rewards.index({step, Slice(), Slice()}) +
        gamma * next_value * next_is_not_terminal -
        this->transitions_.values.index({step, Slice(), Slice()});
    advantage = delta + gamma * lambda * advantage * next_is_not_terminal;
    this->transitions_.rewards.index_put_(
        {step, Slice(), Slice()},
        advantage + this->transitions_.values.index({step, Slice(), Slice()}));
    next_value = this->transitions_.values.index({step, Slice(), Slice()});
  }
  this->transitions_.advantages.index_put_(
      {Slice(), Slice(), Slice()},
      this->transitions_.rewards - this->transitions_.values);
  this->transitions_.advantages.index_put_(
      {Slice(), Slice(), Slice()},
      (this->transitions_.advantages - this->transitions_.advantages.mean()) /
          (this->transitions_.advantages.std() + 1e-8));
}

std::vector<Transition> RolloutStorage::compute_batches(
    const unsigned int &num_batches, const unsigned int &epochs) const {
  unsigned int batch_size = this->cfg_.runner_cfg.num_envs *
                            this->cfg_.runner_cfg.num_steps_per_env /
                            num_batches;
  Tensor indices = torch::randperm(batch_size).to(this->device_);

  DictTensor flatten_kl_params;

  for (const auto &kl_param : this->transitions_.kl_params)
    flatten_kl_params[kl_param.first] = kl_param.second.flatten(0, 1);

  Transition flatten_transitions{
      .actor_obs = this->transitions_.actor_obs.flatten(0, 1),
      .critic_obs = this->transitions_.critic_obs.flatten(0, 1),
      .actions = this->transitions_.actions.flatten(0, 1),
      .rewards = this->transitions_.rewards.flatten(0, 1),
      .advantages = this->transitions_.advantages.flatten(0, 1),
      .dones = this->transitions_.dones.flatten(0, 1),
      .values = this->transitions_.values.flatten(0, 1),
      .log_probs = this->transitions_.log_probs.flatten(0, 1),
      .kl_params = flatten_kl_params};

  std::vector<Transition> batches;
  for (unsigned int epoch = 0; epoch < epochs; epoch++) {
    for (int i = 0; i < num_batches; i++) {
      int start = i * batch_size;
      int end = (i + 1) * batch_size;
      Tensor batch_indices = indices.index({Slice(start, end)});

      DictTensor batch_kl_params;

      for (const auto &kl_param : flatten_transitions.kl_params)
        batch_kl_params[kl_param.first] =
            kl_param.second.index({batch_indices, Slice()});

      Transition batch{
          .actor_obs =
              flatten_transitions.actor_obs.index({batch_indices, Slice()}),
          .critic_obs =
              flatten_transitions.critic_obs.index({batch_indices, Slice()}),
          .actions =
              flatten_transitions.actions.index({batch_indices, Slice()}),
          .rewards =
              flatten_transitions.rewards.index({batch_indices, Slice()}),
          .advantages =
              flatten_transitions.advantages.index({batch_indices, Slice()}),
          .dones = flatten_transitions.dones.index({batch_indices, Slice()}),
          .values = flatten_transitions.values.index({batch_indices, Slice()}),
          .log_probs =
              flatten_transitions.log_probs.index({batch_indices, Slice()}),
          .kl_params = batch_kl_params};
      batches.push_back(batch);
    }
  }
  return batches;
}

}  // namespace storage