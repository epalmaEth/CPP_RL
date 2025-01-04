#pragma once

#include <torch/torch.h>

#include "configs/configs.h"
#include "utils/types.h"

namespace modules {

using MLPCfg = configs::MLPCfg;

class MLP : public torch::nn::Module {
 public:
  MLP(const MLPCfg& cfg);

  Tensor forward(const Tensor& x) { return this->network_->forward(x); }

 private:
  void add_activation_(const std::string& activation);
  NN network_ = nullptr;
};

}  // namespace modules