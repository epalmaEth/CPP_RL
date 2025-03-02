#pragma once

#include "modules/distribution.h"
#include "utils/types.h"

namespace modules {

class Beta : public Distribution {
 public:
  explicit Beta(const configs::DistributionCfg& cfg)
    : Distribution(cfg),
      alpha_(torch::ones({cfg.num_inputs})),
      beta_(torch::ones({cfg.num_inputs})),
      min_(cfg.action_min),
      max_(cfg.action_max) {
    this->register_buffer("alpha", this->alpha_);
    this->register_buffer("beta", this->beta_);
  }

  ~Beta() override = default;

  void update(const Tensor& hidden_output) override;
  const Tensor sample() const override {
    return this->scale_(
      at::_sample_dirichlet(torch::stack({this->alpha_, this->beta_}, /*dim=*/-1)));
  }
  const Tensor get_mode() const override {
    return this->scale_((this->alpha_ - 1.f) / (this->alpha_ + this->beta_ - 2.f));
  }
  const Tensor get_log_prob(const Tensor& actions) const override;
  const Tensor get_entropy() const override;
  const Tensor get_kl(const DictTensor& old_kl_params) const override;
  const DictTensor get_kl_params() const override;

 private:
  const Tensor scale_(const Tensor& actions) const {
    return this->min_ + (this->max_ - this->min_) * actions;
  }
  const Tensor unscale_(const Tensor& actions) const {
    return (actions - this->min_) / (this->max_ - this->min_);
  }
  Tensor alpha_;
  Tensor beta_;
  const Tensor max_;
  const Tensor min_;
};

}  // namespace modules