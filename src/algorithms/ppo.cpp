#include "algorithms/ppo.h"

#include <torch/torch.h>

namespace algorithms {

PPO::PPO(const configs::CfgPointer& cfg, const Device& device) : cfg_(cfg), device_(device) {
  this->actor_critic_ = std::make_unique<modules::ActorCritic>(cfg->actor_cfg, cfg->critic_cfg);
  this->rollout_storage_ = std::make_unique<storage::RolloutStorage>(cfg, device);

  auto options = torch::optim::AdamOptions(cfg->ppo_cfg.learning_rate);
  this->optimizer_ =
    std::make_unique<torch::optim::Adam>(this->actor_critic_->parameters(), options);

  this->initialize_();
}

void PPO::act(Tensor& actions, const Tensor& actor_obs, const Tensor& critic_obs) {
  this->transition_.actor_obs.copy_(actor_obs);
  this->transition_.critic_obs.copy_(critic_obs);
  this->transition_.actions.copy_(this->actor_critic_->forward(actor_obs).detach());
  this->transition_.values.copy_(this->actor_critic_->evaluate(critic_obs).detach());
  this->transition_.log_probs.copy_(
    this->actor_critic_->get_actions_log_prob(this->transition_.actions).detach());
  for (const auto& [key, value] : this->actor_critic_->get_distribution_kl_params())
    this->transition_.kl_params[key].copy_(value.detach());

  actions.copy_(this->transition_.actions);
}

void PPO::process_step(const Tensor& rewards, const Tensor& terminated, const Tensor& truncated) {
  const Tensor& bootstrapped_rewards =
    rewards + this->cfg_->ppo_cfg.gamma * this->transition_.values.squeeze(1) * truncated;
  const Tensor& done = terminated | truncated;
  this->transition_.rewards.copy_(bootstrapped_rewards.view({-1, 1}));
  this->transition_.dones.copy_(done.view({-1, 1}));

  this->rollout_storage_->push_back(this->transition_);
}

void PPO::compute_returns(const Tensor critic_obs) {
  const Tensor& last_values = this->actor_critic_->evaluate(critic_obs).detach();
  this->rollout_storage_->compute_advantage(last_values, this->cfg_->ppo_cfg.gamma,
                                            this->cfg_->ppo_cfg.lam);
}

const LossMetrics PPO::update_actor_critic() {
  Tensor actor_loss = torch::zeros({1}, this->device_);
  Tensor critic_loss = torch::zeros({1}, this->device_);
  Tensor entropy_loss = torch::zeros({1}, this->device_);
  Tensor kl_loss = torch::zeros({1}, this->device_);

  std::vector<storage::Transition> batches;
  this->rollout_storage_->update_batches(batches);

  for (const storage::Transition& batch : batches) {
    this->actor_critic_->forward(batch.actor_obs);
    const Tensor& new_log_probs = this->actor_critic_->get_actions_log_prob(batch.actions);
    const Tensor& new_values = this->actor_critic_->evaluate(batch.critic_obs);
    const Tensor& entropy = this->actor_critic_->get_entropy().mean();

    {
      torch::NoGradGuard no_grad;
      float kl = this->actor_critic_->get_kl(batch.kl_params).mean().item<float>();
      kl_loss += kl;

      if (this->cfg_->ppo_cfg.learning_rate_schedule == "adaptive") {
        float learning_rate = this->get_learning_rate();
        if (kl > 2.0f * this->cfg_->ppo_cfg.desired_kl)
          learning_rate = std::max(this->cfg_->ppo_cfg.min_learning_rate, learning_rate / 1.5f);
        else if (kl < 0.5f * this->cfg_->ppo_cfg.desired_kl && kl >= 0.0f)
          learning_rate = std::min(this->cfg_->ppo_cfg.max_learning_rate, learning_rate * 1.5f);
        this->optimizer_->param_groups()[0].options().set_lr(learning_rate);
      }
    }

    const Tensor& ratio = torch::exp(new_log_probs - batch.log_probs);
    const Tensor& surrogate = -ratio * batch.advantages;
    const Tensor& surrogate_clipped =
      -batch.advantages *
      ratio.clamp(1.0f - this->cfg_->ppo_cfg.clip_param, 1.0f + this->cfg_->ppo_cfg.clip_param);
    const Tensor& worst_mean_surrogate_loss = torch::max(surrogate, surrogate_clipped).mean();

    Tensor worst_mean_value_loss;
    if (this->cfg_->ppo_cfg.use_clipped_value_loss) {
      const Tensor& value_clipped =
        batch.values + (new_values - batch.values)
                         .clamp(-this->cfg_->ppo_cfg.clip_param, this->cfg_->ppo_cfg.clip_param);
      const Tensor& value_loss = (new_values - batch.rewards).pow(2);
      const Tensor& value_clipped_loss = (value_clipped - batch.rewards).pow(2);
      worst_mean_value_loss = torch::max(value_loss, value_clipped_loss).mean();
    } else
      worst_mean_value_loss = (new_values - batch.rewards).pow(2).mean();

    const Tensor& loss = worst_mean_surrogate_loss +
                         this->cfg_->ppo_cfg.value_loss_coef * worst_mean_value_loss -
                         this->cfg_->ppo_cfg.entropy_coef * entropy;

    this->optimizer_->zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_norm_(this->actor_critic_->parameters(),
                                      this->cfg_->ppo_cfg.max_grad_norm);
    this->optimizer_->step();

    {
      torch::NoGradGuard no_grad;
      actor_loss += worst_mean_surrogate_loss;
      critic_loss += worst_mean_value_loss;
      entropy_loss += entropy;
    }
  }

  this->rollout_storage_->clear();

  unsigned int num_batches = this->cfg_->ppo_cfg.num_batches * this->cfg_->ppo_cfg.num_epochs;
  return LossMetrics(actor_loss.item<float>() / num_batches,
                     critic_loss.item<float>() / num_batches,
                     entropy_loss.item<float>() / num_batches, kl_loss.item<float>() / num_batches);
}

void PPO::save_models(torch::serialize::OutputArchive& archive) const {
  torch::serialize::OutputArchive actor_critic_archive;
  this->actor_critic_->save(actor_critic_archive);
  archive.write("actor_critic", actor_critic_archive);

  torch::serialize::OutputArchive optimizer_archive;
  this->optimizer_->save(optimizer_archive);
  archive.write("optimizer", optimizer_archive);
}

void PPO::load_models(torch::serialize::InputArchive& archive, const bool& load_optimizer) {
  torch::serialize::InputArchive actor_critic_archive;
  archive.read("actor_critic", actor_critic_archive);
  this->actor_critic_->load(actor_critic_archive);

  if (load_optimizer) {
    torch::serialize::InputArchive optimizer_archive;
    archive.read("optimizer", optimizer_archive);
    this->optimizer_->load(optimizer_archive);
  }
}

void PPO::initialize_() {
  DictTensor zero_kl_params;
  int num_envs = this->cfg_->env_cfg.num_envs;
  const DictTensor& kl_params = this->actor_critic_->get_distribution_kl_params();

  for (const auto& [key, value] : kl_params)
    zero_kl_params[key] = torch::zeros({num_envs, value.size(0)}, this->device_);

  int actor_obs_size = this->cfg_->actor_cfg.mlp_cfg.num_inputs;
  int critic_obs_size = this->cfg_->critic_cfg.mlp_cfg.num_inputs;
  int action_size = this->cfg_->actor_cfg.mlp_cfg.num_outputs;

  this->transition_.actor_obs = torch::zeros({num_envs, actor_obs_size}, this->device_);
  this->transition_.critic_obs = torch::zeros({num_envs, critic_obs_size}, this->device_);
  this->transition_.actions = torch::zeros({num_envs, action_size}, this->device_);
  this->transition_.rewards = torch::zeros({num_envs, 1}, this->device_);
  this->transition_.dones = torch::zeros({num_envs, 1}, this->device_);
  this->transition_.values = torch::zeros({num_envs, 1}, this->device_);
  this->transition_.log_probs = torch::zeros({num_envs, 1}, this->device_);
  this->transition_.kl_params = zero_kl_params;

  this->rollout_storage_->initialize(zero_kl_params);
  this->actor_critic_->to(this->device_);
}

}  // namespace algorithms