#pragma once

#include "configs/configs.h"
#include "utils/types.h"

namespace modules {

class Normalizer : public NNModule {
 public:
  Normalizer() = default;

  ~Normalizer() = default;

  virtual const Tensor forward(const Tensor& observations) = 0;
  virtual const Tensor normalize(const Tensor& observations) const = 0;
  virtual const Tensor denormalize(const Tensor& observations) const = 0;

  void train() { this->inference_mode_ = false; }
  void eval() { this->inference_mode_ = true; }

 protected:
  bool inference_mode_ = false;
};

using NormalizerPointer = std::shared_ptr<Normalizer>;

}  // namespace modules