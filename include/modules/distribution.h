#pragma once

#include "configs/configs.h"
#include "utils/types.h"

namespace modules {

using DistributionCfg = configs::DistributionCfg;

class Distribution : public NNModule {
 public:
  Distribution(const DistributionCfg& cfg)
      : std_(torch::full({cfg.num_inputs}, cfg.init_noise_std)),
        mean_(torch::zeros({cfg.num_inputs})) {
    this->register_parameter("std", this->std_);
    this->register_buffer("mean", this->mean_);
  }

  virtual ~Distribution() = default;

  virtual void update(const Tensor& hidden_output) = 0;
  virtual Tensor sample() const = 0;
  virtual Tensor get_mean() const { return this->mean_; }
  virtual Tensor get_mode() const = 0;
  virtual Tensor get_std() const { return this->std_; }
  virtual Tensor get_log_prob(const Tensor& actions) const = 0;
  virtual Tensor get_entropy() const = 0;
  virtual Tensor get_kl(const DictTensor& old_kl_params) const = 0;
  virtual DictTensor get_kl_params() const = 0;

 protected:
  Tensor std_;
  Tensor mean_;
};

class Normal : public Distribution {
 public:
  // Inherit constructors from the Distribution class
  using Distribution::Distribution;

  ~Normal() override = default;

  void update(const Tensor& hidden_output) override;
  Tensor sample() const override;
  Tensor get_mode() const override { return this->mean_; }
  Tensor get_log_prob(const Tensor& actions) const override;
  Tensor get_entropy() const override;
  Tensor get_kl(const DictTensor& old_kl_params) const override;
  DictTensor get_kl_params() const override;
};

class Beta : public Distribution {
 public:
  Beta(const DistributionCfg& cfg)
      : Distribution(cfg),
        alpha_(torch::ones({cfg.num_inputs})),
        beta_(torch::ones({cfg.num_inputs})) {
    this->register_buffer("alpha", this->alpha_);
    this->register_buffer("beta", this->beta_);
  }

  ~Beta() override = default;

  void update(const Tensor& hidden_output) override;
  Tensor sample() const override;
  Tensor get_mode() const override {
    return (this->alpha_ - 1.) / (this->alpha_ + this->beta_ - 2.);
  }
  Tensor get_log_prob(const Tensor& actions) const override;
  Tensor get_entropy() const override;
  Tensor get_kl(const DictTensor& old_kl_params) const override;
  DictTensor get_kl_params() const override;

 private:
  Tensor alpha_;
  Tensor beta_;
};

class DistributionFactory {
 public:
  static std::shared_ptr<Distribution> create(const DistributionCfg& cfg) {
    if (cfg.type == "normal") return std::make_shared<Normal>(cfg);
    if (cfg.type == "beta") return std::make_shared<Beta>(cfg);
    throw std::invalid_argument("Unknown distribution type: " + cfg.type);
  }
};
}  // namespace modules