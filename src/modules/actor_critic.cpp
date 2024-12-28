#include "modules/actor_critic.h"

#include <iostream>

#include "utils/utils.h"

namespace modules {

Actor::Actor(const int& num_observations, const int& num_actions,
             const int& depth, const double& init_noise_std,
             const torch::nn::Functional& activation,
             const string& normalizer_type, const string& distribution_type,
             const Device& device) {
  this->normalizer_ =
      NormalizerFactory::create(normalizer_type, num_observations, device);

  const int& width = utils::next_power_of_2(num_observations);
  this->network_ = utils::create_MLP(num_observations, num_actions, width,
                                     depth, activation);

  this->distribution_ = DistributionFactory::create(
      distribution_type, num_actions, init_noise_std, device);

  this->network_->to(device);
  this->register_module("network", this->network_);
  this->register_module("distribution", this->distribution_);

  std::cout << "Actor MLP: " << this->network_ << std::endl;
}

Tensor Actor::forward(const Tensor& actor_observations) {
  this->distribution_->update(
      this->network_->forward(this->normalizer_->forward(actor_observations)));
  if (this->inference_mode_) return this->distribution_->get_mode();
  return this->distribution_->get_sample();
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

Critic::Critic(const int& num_observations, const int& depth,
               const torch::nn::Functional& activation,
               const string& normalizer_type, const Device& device) {
  this->normalizer_ =
      NormalizerFactory::create(normalizer_type, num_observations, device);

  const int& width = utils::next_power_of_2(num_observations);
  this->network_ =
      utils::create_MLP(num_observations, 1, width, depth, activation);

  this->network_->to(device);
  this->register_module("network", this->network_);

  std::cout << "Critic MLP: " << this->network_ << std::endl;
}

std::function<Tensor(const Tensor&)> ActorCritic::get_inference_policy() {
  this->eval();
  return [this](const Tensor& actor_observations) {
    return this->forward(actor_observations);
  };
}
}  // namespace modules