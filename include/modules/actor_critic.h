#pragma once

#include "configs/configs.h"
#include "distribution.h"
#include "mlp.h"
#include "normalizer.h"
#include "utils/types.h"

namespace modules {

using NormalizerPointer = std::shared_ptr<Normalizer>;
using MLPPointer = std::shared_ptr<MLP>;
using DistributionPointer = std::shared_ptr<Distribution>;
using ActorCfg = configs::ActorCfg;
using CriticCfg = configs::CriticCfg;

class Actor : public NNModule {
 public:
  Actor(const ActorCfg& cfg);

  Tensor forward(const Tensor& actor_obs);
  Tensor forward_inference(const Tensor& actor_obs);
  Tensor get_mean() const { return this->distribution_->get_mean(); }
  Tensor get_std() const { return this->distribution_->get_std(); }
  Tensor get_log_prob(const Tensor& actions) const {
    return this->distribution_->get_log_prob(actions);
  }
  Tensor get_entropy() const { return this->distribution_->get_entropy(); }
  Tensor get_kl(const DictTensor& old_kl_params) const {
    return this->distribution_->get_kl(old_kl_params);
  }
  DictTensor get_kl_params() const {
    return this->distribution_->get_kl_params();
  }
  void train();
  void eval();

 private:
  bool inference_mode_ = false;
  NormalizerPointer normalizer_ = nullptr;
  MLPPointer network_ = nullptr;
  DistributionPointer distribution_ = nullptr;
};

class Critic : public NNModule {
 public:
  Critic(const CriticCfg& cfg);

  Tensor forward(const Tensor& critic_obs) {
    return this->network_->forward(this->normalizer_->forward(critic_obs));
  };

 private:
  NormalizerPointer normalizer_ = nullptr;
  MLPPointer network_ = nullptr;
};

using ActorPointer = std::shared_ptr<Actor>;
using CriticPointer = std::shared_ptr<Critic>;

class ActorCritic : public NNModule {
 public:
  ActorCritic(const ActorCfg& actor_cfg, const CriticCfg& critic_cfg);

  Tensor forward(const Tensor& actor_obs) {
    return this->actor_->forward(actor_obs);
  }
  Tensor evaluate(const Tensor& critic_obs) {
    return this->critic_->forward(critic_obs);
  }
  Tensor get_action_mean() const { return this->actor_->get_mean(); }
  Tensor get_action_std() const { return this->actor_->get_std(); }
  Tensor get_actions_log_prob(const Tensor& actions) const {
    return this->actor_->get_log_prob(actions).sum(/*dim=*/-1,
                                                   /*keepdim=*/true);
  }
  Tensor get_entropy() const { return this->actor_->get_entropy().sum(-1); }
  Tensor get_kl(const DictTensor& old_kl_params) const {
    return this->actor_->get_kl(old_kl_params);
  }
  DictTensor get_distribution_kl_params() const {
    return this->actor_->get_kl_params();
  }
  std::function<Tensor(const Tensor&)> get_inference_policy() const {
    return [this](const Tensor& actor_obs) {
      return this->actor_->forward_inference(actor_obs);
    };
  }
  void train() { this->actor_->train(); }
  void eval() { this->actor_->eval(); }

 private:
  ActorPointer actor_ = nullptr;
  CriticPointer critic_ = nullptr;
};
}  // namespace modules