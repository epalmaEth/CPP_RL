#pragma once

#include "utils/types.h"

namespace modules {
class Distribution : public NNModule {
 public:
  Distribution(const int& input_size, const double& init_noise_std,
               const Device& device) {
    this->std_ = torch::full({input_size}, init_noise_std).to(device);

    this->register_parameter("std", this->std_);
  }

  virtual ~Distribution() = default;

  virtual void update(const Tensor& hidden_output) = 0;
  virtual Tensor get_sample() const = 0;
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
  Tensor get_sample() const override;
  Tensor get_mode() const override { return this->mean_; }
  Tensor get_log_prob(const Tensor& actions) const override;
  Tensor get_entropy() const override;
  Tensor get_kl(const DictTensor& old_kl_params) const override;
  DictTensor get_kl_params() const override;
};

class Beta : public Distribution {
 public:
  Beta(const int& input_size, const double& init_noise_std,
       const Device& device)
      : Distribution(input_size, init_noise_std, device) {
    this->alpha_ = torch::full({input_size}, 1.).to(device);
    this->beta_ = torch::full({input_size}, 1.).to(device);
  }

  ~Beta() override = default;

  void update(const Tensor& hidden_output) override;
  Tensor get_sample() const override;
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
}  // namespace modules