#pragma once

#include "configs/configs.h"
#include "distribution.h"
#include "mlp.h"
#include "normalizer.h"
#include "utils/types.h"

namespace modules {

class Actor : public NNModule {
 public:
  explicit Actor(const configs::ActorCfg& cfg);

  const Tensor forward(const Tensor& actor_obs);
  const Tensor forward_inference(const Tensor& actor_obs);
  const Tensor& get_mean() const { return this->distribution_->get_mean(); }
  const Tensor& get_std() const { return this->distribution_->get_std(); }
  const Tensor get_log_prob(const Tensor& actions) const {
    return this->distribution_->get_log_prob(actions);
  }
  const Tensor get_entropy() const { return this->distribution_->get_entropy(); }
  const Tensor get_kl(const DictTensor& old_kl_params) const {
    return this->distribution_->get_kl(old_kl_params);
  }
  const DictTensor get_kl_params() const { return this->distribution_->get_kl_params(); }
  void train();
  void eval();

 private:
  bool inference_mode_ = false;
  NormalizerPointer normalizer_;
  MLPPointer network_;
  DistributionPointer distribution_;
};

class Critic : public NNModule {
 public:
  explicit Critic(const configs::CriticCfg& cfg);

  const Tensor forward(const Tensor& critic_obs) {
    return this->network_->forward(this->normalizer_->forward(critic_obs));
  }

 private:
  NormalizerPointer normalizer_;
  MLPPointer network_;
};

using ActorPointer = std::shared_ptr<Actor>;
using CriticPointer = std::shared_ptr<Critic>;

class ActorCritic : public NNModule {
 public:
  ActorCritic(const configs::ActorCfg& actor_cfg, const configs::CriticCfg& critic_cfg);

  const Tensor forward(const Tensor& actor_obs) { return this->actor_->forward(actor_obs); }
  const Tensor evaluate(const Tensor& critic_obs) { return this->critic_->forward(critic_obs); }
  const Tensor& get_action_std() const { return this->actor_->get_std(); }
  const Tensor get_actions_log_prob(const Tensor& actions) const {
    return this->actor_->get_log_prob(actions).sum(/*dim=*/-1,
                                                   /*keepdim=*/true);
  }
  const Tensor get_entropy() const { return this->actor_->get_entropy().sum(/*dim=*/-1); }
  const Tensor get_kl(const DictTensor& old_kl_params) const {
    return this->actor_->get_kl(old_kl_params).sum(/*dim=*/-1);
  }
  const DictTensor get_distribution_kl_params() const { return this->actor_->get_kl_params(); }
  const std::function<Tensor(const Tensor&)> get_inference_policy() const {
    return [this](const Tensor& actor_obs) { return this->actor_->forward_inference(actor_obs); };
  }
  void train() { this->actor_->train(); }
  void eval() { this->actor_->eval(); }

 private:
  ActorPointer actor_;
  CriticPointer critic_;
};

using ActorCriticPointer = std::unique_ptr<modules::ActorCritic>;
}  // namespace modules