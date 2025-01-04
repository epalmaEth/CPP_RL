#include "algorithms/ppo.h"

#include <torch/torch.h>

namespace algorithms {

PPO::PPO(const Cfg &cfg, const Device &device) : cfg_(cfg) {
  this->actor_critic_ =
      std::make_unique<modules::ActorCritic>(cfg.actor_cfg, cfg.critic_cfg);
  this->optimizer_ = std::make_unique<torch::optim::Adam>(
      this->actor_critic_->parameters(),
      torch::optim::AdamOptions(cfg.ppo_cfg.learning_rate));
  this->rollout_storage_ =
      std::make_unique<storage::RolloutStorage>(cfg, device);

  this->initialize_(device);
}

Tensor PPO::act(const Tensor &actor_obs, const Tensor &critic_obs) {
  this->transition_.actor_obs.index_put_({Slice(), Slice()}, actor_obs);
  this->transition_.critic_obs.index_put_({Slice(), Slice()}, critic_obs);
  this->transition_.actions.index_put_(
      {Slice(), Slice()}, this->actor_critic_->forward(actor_obs).detach());
  this->transition_.values.index_put_(
      {Slice(), Slice()}, this->actor_critic_->evaluate(critic_obs).detach());
  this->transition_.log_probs.index_put_(
      {Slice(), Slice()},
      this->actor_critic_->get_actions_log_prob(this->transition_.actions)
          .detach());
  for (const auto &kl_param : this->actor_critic_->get_distribution_kl_params())
    this->transition_.kl_params[kl_param.first].index_put_(
        {Slice(), Slice()}, kl_param.second.detach());

  return this->transition_.actions;
}

void PPO::process_step(const Tensor &rewards, const Tensor &terminated,
                       const Tensor &truncated) {
  const Tensor &bootstrapped_rewards =
      rewards + this->cfg_.ppo_cfg.gamma * this->transition_.values.squeeze(1) *
                    truncated;
  const Tensor &done = terminated | truncated;
  this->transition_.rewards.index_put_(
      {Slice(), Slice()}, bootstrapped_rewards.reshape({-1, 1}).detach());
  this->transition_.dones.index_put_({Slice(), Slice()},
                                     done.reshape({-1, 1}).detach());

  this->rollout_storage_->insert(this->transition_);
}

void PPO::compute_returns(const Tensor critic_obs) {
  const Tensor &last_values =
      this->actor_critic_->evaluate(critic_obs).detach();
  this->rollout_storage_->compute_cumulative_rewards(
      last_values, this->cfg_.ppo_cfg.gamma, this->cfg_.ppo_cfg.lam);
}

std::pair<Tensor, Tensor> PPO::update_actor_critic() {
  Tensor mean_value_loss =
      torch::zeros({1}).to(this->transition_.actor_obs.device());
  Tensor mean_surrogate_loss =
      torch::zeros({1}).to(this->transition_.actor_obs.device());

  const std::vector<Transition> &batches =
      this->rollout_storage_->compute_batches(this->cfg_.ppo_cfg.num_batches,
                                              this->cfg_.ppo_cfg.num_epochs);

  for (const Transition &batch : batches) {
    this->actor_critic_->forward(batch.actor_obs);
    const Tensor &new_log_probs =
        this->actor_critic_->get_actions_log_prob(batch.actions);
    const Tensor &new_values = this->actor_critic_->evaluate(batch.critic_obs);
    const Tensor entropy = this->actor_critic_->get_entropy();

    if (this->cfg_.ppo_cfg.learning_rate_schedule == "adaptive") {
      torch::NoGradGuard no_grad;
      const float &kl =
          this->actor_critic_->get_kl(batch.kl_params).mean().item<float>();
      float learning_rate =
          this->optimizer_->param_groups()[0].options().get_lr();
      if (kl > 2.0 * this->cfg_.ppo_cfg.desired_kl)
        learning_rate = std::max(1e-05, learning_rate / 1.5);
      else if (kl < 0.5 * this->cfg_.ppo_cfg.desired_kl && kl >= 0.0)
        learning_rate = std::min(1e-02, learning_rate * 1.5);
      this->optimizer_->param_groups()[0].options().set_lr(learning_rate);
    }

    const Tensor &ratio = torch::exp(new_log_probs - batch.log_probs);
    const Tensor &surrogate = -ratio * batch.advantages;
    const Tensor &surrogate_clipped =
        -batch.advantages * ratio.clamp(1.0 - this->cfg_.ppo_cfg.clip_param,
                                        1.0 + this->cfg_.ppo_cfg.clip_param);
    const Tensor &worst_mean_surrogate_loss =
        torch::max(surrogate, surrogate_clipped).mean();

    Tensor worst_mean_value_loss;
    if (this->cfg_.ppo_cfg.use_clipped_value_loss) {
      const Tensor &value_clipped =
          batch.values + (new_values - batch.values)
                             .clamp(-this->cfg_.ppo_cfg.clip_param,
                                    this->cfg_.ppo_cfg.clip_param);
      const Tensor &value_loss = (new_values - batch.rewards).pow(2);
      const Tensor &value_clipped_loss = (value_clipped - batch.rewards).pow(2);
      worst_mean_value_loss = torch::max(value_loss, value_clipped_loss).mean();
    } else {
      worst_mean_value_loss = (new_values - batch.rewards).pow(2).mean();
    }

    const Tensor &loss =
        worst_mean_surrogate_loss +
        this->cfg_.ppo_cfg.value_loss_coef * worst_mean_value_loss -
        this->cfg_.ppo_cfg.entropy_coef * entropy;

    this->optimizer_->zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_norm_(this->actor_critic_->parameters(),
                                      this->cfg_.ppo_cfg.max_grad_norm);
    this->optimizer_->step();

    mean_value_loss += worst_mean_value_loss;
    mean_surrogate_loss += worst_mean_surrogate_loss;
  }

  const int &num_updates =
      this->cfg_.ppo_cfg.num_batches * this->cfg_.ppo_cfg.num_epochs;
  mean_value_loss /= num_updates;
  mean_surrogate_loss /= num_updates;

  this->rollout_storage_->clear();
  return std::pair<torch::Tensor, torch::Tensor>(mean_value_loss,
                                                 mean_surrogate_loss);
}

torch::serialize::OutputArchive PPO::save_models() const {
  torch::serialize::OutputArchive archive;

  torch::serialize::OutputArchive actor_critic_archive;
  this->actor_critic_->save(actor_critic_archive);
  archive.write("actor_critic", actor_critic_archive);

  torch::serialize::OutputArchive optimizer_archive;
  this->optimizer_->save(optimizer_archive);
  archive.write("optimizer", optimizer_archive);

  return archive;
}

void PPO::load_models(torch::serialize::InputArchive &archive,
                      const bool &load_optimizer) {
  torch::serialize::InputArchive actor_critic_archive;
  archive.read("actor_critic", actor_critic_archive);
  this->actor_critic_->load(actor_critic_archive);

  if (load_optimizer) {
    torch::serialize::InputArchive optimizer_archive;
    archive.read("optimizer", optimizer_archive);
    this->optimizer_->load(optimizer_archive);
  }
}

void PPO::initialize_(const Device &device) {
  DictTensor zero_kl_params;
  const int &num_envs = this->cfg_.runner_cfg.num_envs;
  const DictTensor &kl_params =
      this->actor_critic_->get_distribution_kl_params();

  for (const auto &kl_param : kl_params)
    zero_kl_params[kl_param.first] =
        torch::zeros({num_envs, kl_param.second.size(0)}).to(device);

  const int &actor_obs_size = this->cfg_.actor_cfg.mlp_cfg.num_inputs;
  const int &critic_obs_size = this->cfg_.critic_cfg.mlp_cfg.num_inputs;
  const int &action_size = this->cfg_.actor_cfg.mlp_cfg.num_outputs;

  this->transition_.actor_obs =
      torch::zeros({num_envs, actor_obs_size}).to(device);
  this->transition_.critic_obs =
      torch::zeros({num_envs, critic_obs_size}).to(device);
  this->transition_.actions = torch::zeros({num_envs, action_size}).to(device);
  this->transition_.rewards = torch::zeros({num_envs, 1}).to(device);
  this->transition_.dones = torch::zeros({num_envs, 1}).to(device);
  this->transition_.values = torch::zeros({num_envs, 1}).to(device);
  this->transition_.log_probs = torch::zeros({num_envs, 1}).to(device);
  this->transition_.kl_params = zero_kl_params;

  this->rollout_storage_->initialize(zero_kl_params);
  this->actor_critic_->to(device);
}

}  // namespace algorithms