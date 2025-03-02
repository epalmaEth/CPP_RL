#pragma once

#include "configs/configs.h"
#include "modules/normalizer.h"
#include "utils/types.h"

namespace modules {

class IdentityNormalizer : public Normalizer {
 public:
  // Inherit the constructor from the base class
  using Normalizer::Normalizer;

  const Tensor forward(const Tensor& observations) override { return observations; }
  const Tensor normalize(const Tensor& observations) const override { return observations; }
  const Tensor denormalize(const Tensor& observations) const override { return observations; }
};

}  // namespace modules