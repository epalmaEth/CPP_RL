#include "modules/normalizer.h"

namespace modules {

Tensor EmpiricalNormalizer::forward(const Tensor& observations) {
  if (!this->inference_mode_) this->update(observations);
  return this->normalize(observations);
}

void EmpiricalNormalizer::update(const Tensor& observations) {
  this->count_ += observations.size(0);

  const double rate = static_cast<double>(observations.size(0)) / this->count_;
  const Tensor& var_obs =
      torch::var(observations, /*dim=*/0, /*unbiased=*/true);
  const Tensor& mean_obs = torch::mean(observations, /*dim=*/0);
  const Tensor& delta_mean = mean_obs - this->mean_;

  this->mean_ += rate * delta_mean;
  this->var_ += rate * (var_obs - this->var_ + delta_mean.square());
}

Tensor EmpiricalNormalizer::normalize(const Tensor& observations) const {
  const Tensor& mean = this->mean_.expand_as(observations);
  const Tensor& std = this->var_.sqrt().expand_as(observations);
  return (observations - mean) / (std + 1e-07);
}

Tensor EmpiricalNormalizer::denormalize(const Tensor& observations) const {
  const Tensor& mean = this->mean_.expand_as(observations);
  const Tensor& std = this->var_.sqrt().expand_as(observations);
  return mean + observations * (std + 1e-07);
}

}  // namespace modules
