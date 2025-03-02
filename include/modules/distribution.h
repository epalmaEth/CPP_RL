#pragma once

#include "configs/configs.h"
#include "utils/types.h"

namespace modules {

class Distribution : public NNModule {
 public:
  Distribution(const configs::DistributionCfg& cfg)
    : std_(torch::full({cfg.num_inputs}, cfg.init_noise_std)),
      mean_(torch::zeros({cfg.num_inputs})) {
    this->register_parameter("std", this->std_);
    this->register_buffer("mean", this->mean_);
  }

  virtual ~Distribution() = default;

  virtual void update(const Tensor& hidden_output) = 0;
  virtual const Tensor sample() const = 0;
  virtual const Tensor& get_mean() const { return this->mean_; }
  virtual const Tensor get_mode() const = 0;
  virtual const Tensor& get_std() const { return this->std_; }
  virtual const Tensor get_log_prob(const Tensor& actions) const = 0;
  virtual const Tensor get_entropy() const = 0;
  virtual const Tensor get_kl(const DictTensor& old_kl_params) const = 0;
  virtual const DictTensor get_kl_params() const = 0;

 protected:
  Tensor std_;
  Tensor mean_;
};

using DistributionPointer = std::shared_ptr<Distribution>;
}  // namespace modules