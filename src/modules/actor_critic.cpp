#include "modules/actor_critic.h"

#include <iostream>

#include "utils/utils.h"

namespace modules {

Actor::Actor(const int& num_observations, const int& num_actions,
             const int& depth, const double& init_noise_std,
             const torch::nn::Functional& activation,
             const string& distribution_type, const Device& device) {
  const int& width = utils::next_power_of_2(num_observations);
  this->network_ = utils::create_MLP(num_observations, num_actions, width,
                                     depth, activation);
  if (distribution_type == "normal")
    this->distribution_ =
        std::make_shared<Normal>(num_actions, init_noise_std, device);
  else if (distribution_type == "beta") {
    this->distribution_ =
        std::make_shared<Beta>(num_actions, init_noise_std, device);
    this->network_->push_back(torch::nn::Sigmoid());
  } else
    throw std::invalid_argument("Invalid distribution type");

  this->network_->to(device);
  this->register_module("network", this->network_);
  this->register_module("distribution", this->distribution_);

  std::cout << "Actor MLP: " << this->network_ << std::endl;
}

Tensor Actor::forward(const Tensor& actor_observations, const bool& inference) {
  this->distribution_->update(this->network_->forward(actor_observations));
  if (inference) return this->distribution_->get_mode();
  return this->distribution_->get_sample();
}

Critic::Critic(const int& num_observations, const int& depth,
               const torch::nn::Functional& activation, const Device& device) {
  const int& width = utils::next_power_of_2(num_observations);
  this->network_ =
      utils::create_MLP(num_observations, 1, width, depth, activation);

  this->network_->to(device);
  this->register_module("network", this->network_);

  std::cout << "Critic MLP: " << this->network_ << std::endl;
}
}  // namespace modules