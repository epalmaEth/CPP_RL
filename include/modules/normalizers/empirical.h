#pragma once

#include "configs/configs.h"
#include "modules/normalizer.h"
#include "utils/types.h"

namespace modules {

class EmpiricalNormalizer : public Normalizer {
 public:
  explicit EmpiricalNormalizer(const configs::NormalizerCfg& cfg) {
    this->mean_ = torch::zeros({cfg.num_inputs});
    this->var_ = torch::ones({cfg.num_inputs});

    this->register_buffer("mean", this->mean_);
    this->register_buffer("var", this->var_);
  }

  const Tensor forward(const Tensor& observations) override;
  void update(const Tensor& observations);
  const Tensor normalize(const Tensor& observations) const override;
  const Tensor denormalize(const Tensor& observations) const override;

 private:
  unsigned int count_ = 0;

  Tensor mean_;
  Tensor var_;
};

}  // namespace modules