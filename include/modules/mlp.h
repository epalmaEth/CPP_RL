#pragma once

#include <torch/torch.h>

#include "configs/configs.h"
#include "utils/types.h"

namespace modules {

class MLP : public NNModule {
 public:
  explicit MLP(const configs::MLPCfg& cfg);

  const Tensor forward(const Tensor& x) { return this->network_->forward(x); }

 private:
  void add_activation_(const string& activation);
  NN network_;
};

using MLPPointer = std::shared_ptr<MLP>;
}  // namespace modules