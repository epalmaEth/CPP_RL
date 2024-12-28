#pragma once

#include "distribution.h"
#include "normalizer.h"
#include "utils/types.h"

namespace modules {

class Actor : public NNModule {
 public:
  Actor(const int& num_observations, const int& num_actions, const int& depth,
        const double& init_noise_std, const torch::nn::Functional& activation,
        const string& normalizer_type, const string& distribution_type,
        const Device& device);

  Tensor forward(const Tensor& actor_observations);
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
  std::unique_ptr<Normalizer> normalizer_;
  MLP network_;
  std::shared_ptr<Distribution> distribution_;
};

class Critic : public NNModule {
 public:
  Critic(const int& num_observations, const int& depth,
         const torch::nn::Functional& activation, const string& normalizer_type,
         const Device& device);

  Tensor forward(const Tensor& critic_observations) {
    return this->network_->forward(
        this->normalizer_->forward(critic_observations));
  };

 private:
  std::unique_ptr<Normalizer> normalizer_;
  MLP network_;
};

class ActorCritic : public NNModule {
 public:
  ActorCritic(const int& num_actor_obs, const int& num_critic_obs,
              const int& num_actions, const double& init_noise_std,
              const int& depth_actor, const int& depth_critic,
              const Device& device,
              const torch::nn::Functional& activation_actor,
              const torch::nn::Functional& activation_critic,
              const string& normalizer_type_actor,
              const string& normalizer_type_critic,
              const std::string& distribution_type)
      : actor_(num_actor_obs, num_actions, depth_actor, init_noise_std,
               activation_actor, normalizer_type_actor, distribution_type,
               device),
        critic_(num_critic_obs, depth_critic, activation_critic,
                normalizer_type_critic, device) {}

  Tensor forward(const Tensor& actor_observations) {
    return this->actor_.forward(actor_observations);
  }
  Tensor evaluate(const Tensor& critic_observations) {
    return this->critic_.forward(critic_observations);
  }
  Tensor get_action_mean() const { return this->actor_.get_mean(); }
  Tensor get_action_std() const { return this->actor_.get_std(); }
  Tensor get_actions_log_prob(const Tensor& actions) const {
    return this->actor_.get_log_prob(actions).sum(-1);
  }
  Tensor get_entropy() const { return this->actor_.get_entropy().sum(-1); }
  Tensor get_kl(const DictTensor& old_kl_params) const {
    return this->actor_.get_kl(old_kl_params);
  }
  DictTensor get_distribution_kl_params() const {
    return this->actor_.get_kl_params();
  }
  std::function<Tensor(const Tensor&)> get_inference_policy();
  void train() { this->actor_.train(); }
  void eval() { this->actor_.eval(); }

 private:
  Actor actor_;
  Critic critic_;
};
}  // namespace modules