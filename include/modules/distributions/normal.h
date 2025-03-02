#pragma once

#include <torch/torch.h>

#include "modules/distribution.h"
#include "utils/types.h"

namespace modules {

class Normal : public Distribution {
 public:
  // Inherit constructors from the Distribution class
  using Distribution::Distribution;

  ~Normal() override = default;

  void update(const Tensor& hidden_output) override { this->mean_ = hidden_output; }
  const Tensor sample() const override {
    return at::normal(this->mean_, this->std_.expand_as(this->mean_));
  }
  const Tensor get_mode() const override { return this->mean_; }
  const Tensor get_log_prob(const Tensor& actions) const override {
    const Tensor& var = this->std_.square();
    return -0.5f * (torch::log(2.f * M_PI * var) + (actions - this->mean_).square() / var);
  }
  const Tensor get_entropy() const override {
    return 0.5f * (1.f + torch::log(2.f * M_PI * this->std_.square()));
  }
  const Tensor get_kl(const DictTensor& old_kl_params) const override;
  const DictTensor get_kl_params() const override {
    return {{"mean", this->mean_}, {"std", this->std_}};
  }
};

}  // namespace modules