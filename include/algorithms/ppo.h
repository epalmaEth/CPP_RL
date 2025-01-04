#pragma once

#include <torch/torch.h>

#include "configs/configs.h"
#include "modules/actor_critic.h"
#include "storage/rollout.h"

namespace algorithms {

using Cfg = configs::Cfg;
using ActorCriticPointer = std::unique_ptr<modules::ActorCritic>;
using RolloutStoragePointer = std::unique_ptr<storage::RolloutStorage>;
using AdamPointer = std::unique_ptr<torch::optim::Adam>;
using Transition = storage::Transition;

class PPO {
 public:
  PPO(const Cfg &cfg, const Device &device);

  Tensor act(const Tensor &actor_obs, const Tensor &critic_obs);
  void process_step(const Tensor &rewards, const Tensor &terminated,
                    const Tensor &truncated);
  void compute_returns(const Tensor critic_obs);
  std::pair<Tensor, Tensor> update_actor_critic();
  std::function<Tensor(const Tensor &)> get_inference_policy() const {
    return this->actor_critic_->get_inference_policy();
  }
  Tensor get_action_std() const {
    return this->actor_critic_->get_action_std();
  };
  void train() { this->actor_critic_->train(); };
  void eval() { this->actor_critic_->eval(); };
  torch::serialize::OutputArchive save_models() const;
  void load_models(torch::serialize::InputArchive &archive,
                   const bool &load_optimizer);

 private:
  void initialize_(const Device &device);

  Cfg cfg_;
  ActorCriticPointer actor_critic_;
  AdamPointer optimizer_;
  RolloutStoragePointer rollout_storage_;
  Transition transition_;
};
}  // namespace algorithms