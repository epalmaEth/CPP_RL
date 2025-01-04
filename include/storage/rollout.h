#pragma once

#include "configs/configs.h"
#include "utils/types.h"

namespace storage {

using Cfg = configs::Cfg;

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
};

class RolloutStorage {
 public:
  RolloutStorage(const Cfg &cfg, const Device &device)
      : cfg_(cfg), device_(device) {}

  void initialize(const DictTensor &kl_params);
  void clear() { this->step_ = 0; }
  void insert(const Transition &transition);
  void compute_cumulative_rewards(const Tensor &last_values, const float &gamma,
                                  const float &lambda);
  std::vector<Transition> compute_batches(const unsigned int &num_batches,
                                          const unsigned int &epochs) const;

 private:
  Cfg cfg_;
  Device device_;
  Transition transitions_;

  int step_ = 0;
};

}  // namespace storage