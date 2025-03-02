#include "modules/actor_critic.h"

#include <iostream>

#include "modules/distributions/factory.h"
#include "modules/normalizers/factory.h"
#include "utils/utils.h"

namespace modules {

Actor::Actor(const configs::ActorCfg& cfg) {
  this->normalizer_ = NormalizerFactory::create(cfg.normalizer_cfg);
  this->network_ = std::make_shared<MLP>(cfg.mlp_cfg);
  this->distribution_ = DistributionFactory::create(cfg.distribution_cfg);

  this->register_module("normalizer", this->normalizer_);
  this->register_module("network", this->network_);
  this->register_module("distribution", this->distribution_);
}

const Tensor Actor::forward(const Tensor& actor_obs) {
  this->distribution_->update(this->network_->forward(this->normalizer_->forward(actor_obs)));
  if (this->inference_mode_) return this->distribution_->get_mode();
  return this->distribution_->sample();
}

const Tensor Actor::forward_inference(const Tensor& actor_obs) {
  this->distribution_->update(this->network_->forward(this->normalizer_->forward(actor_obs)));
  return this->distribution_->get_mode();
}

void Actor::train() {
  this->inference_mode_ = false;
  this->normalizer_->train();
  this->network_->train();
}

void Actor::eval() {
  this->inference_mode_ = true;
  this->normalizer_->eval();
  this->network_->eval();
}

Critic::Critic(const configs::CriticCfg& cfg) {
  this->normalizer_ = NormalizerFactory::create(cfg.normalizer_cfg);

  this->network_ = std::make_shared<MLP>(cfg.mlp_cfg);

  this->register_module("normalizer", this->normalizer_);
  this->register_module("network", this->network_);
}

ActorCritic::ActorCritic(const configs::ActorCfg& actor_cfg, const configs::CriticCfg& critic_cfg) {
  this->actor_ = std::make_shared<Actor>(actor_cfg);
  this->critic_ = std::make_shared<Critic>(critic_cfg);

  this->register_module("actor", this->actor_);
  this->register_module("critic", this->critic_);

  std::cout << "Actor: \n" << *this->actor_ << std::endl;
  std::cout << "Critic: \n" << *this->critic_ << std::endl;
}
}  // namespace modules