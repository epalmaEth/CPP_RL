#pragma once

#include "configs/configs.h"
#include "utils/types.h"

namespace storage {

struct Transition {
  Tensor actor_obs;
  Tensor critic_obs;
  Tensor actions;
  Tensor rewards;
  Tensor advantages;
  Tensor dones;
  Tensor values;
  Tensor log_probs;

  DictTensor kl_params;

  friend std::ostream& operator<<(std::ostream& os, const Transition& transition) {
    os << "actor_obs: " << transition.actor_obs.sizes() << std::endl;
    os << "critic_obs: " << transition.critic_obs.sizes() << std::endl;
    os << "actions: " << transition.actions.sizes() << std::endl;
    os << "rewards: " << transition.rewards.sizes() << std::endl;
    os << "advantages: " << transition.advantages.sizes() << std::endl;
    os << "dones: " << transition.dones.sizes() << std::endl;
    os << "values: " << transition.values.sizes() << std::endl;
    os << "log_probs: " << transition.log_probs.sizes() << std::endl;
    os << "kl_params: " << std::endl;
    for (const auto& kl_param : transition.kl_params)
      os << kl_param.first << ": " << kl_param.second.sizes() << std::endl;
    return os;
  }
};

class RolloutStorage {
 public:
  RolloutStorage(const configs::CfgPointer& cfg, const Device& device)
    : cfg_(cfg), device_(device) {}

  void initialize(const DictTensor& kl_params);
  void clear() { this->step_ = 0; }
  void push_back(const Transition& transition);
  void compute_advantage(const Tensor& last_values, const float& gamma, const float& lambda);
  void update_batches(std::vector<storage::Transition>& batches);

 private:
  const configs::CfgPointer cfg_;

  const Device device_;
  Transition transitions_;
  Tensor returns_;
  int step_ = 0;
};

using RolloutStoragePointer = std::unique_ptr<RolloutStorage>;
}  // namespace storage