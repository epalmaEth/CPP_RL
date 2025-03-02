#include "modules/normalizers/empirical.h"

namespace modules {

const Tensor EmpiricalNormalizer::forward(const Tensor& observations) {
  if (!this->inference_mode_) this->update(observations);
  return this->normalize(observations);
}

void EmpiricalNormalizer::update(const Tensor& observations) {
  this->count_ += observations.size(0);

  float rate = static_cast<float>(observations.size(0)) / this->count_;
  const Tensor& var_obs = torch::var(observations, /*dim=*/0, /*unbiased=*/true);
  const Tensor& mean_obs = torch::mean(observations, /*dim=*/0);
  const Tensor& delta_mean = mean_obs - this->mean_;

  this->mean_.add_(rate * delta_mean);
  this->var_.add_(rate * (var_obs - this->var_ + delta_mean.square()));
}

const Tensor EmpiricalNormalizer::normalize(const Tensor& observations) const {
  const Tensor& mean = this->mean_.expand_as(observations);
  const Tensor& std = this->var_.sqrt().expand_as(observations);
  return (observations - mean) / (std + EPS);
}

const Tensor EmpiricalNormalizer::denormalize(const Tensor& observations) const {
  const Tensor& mean = this->mean_.expand_as(observations);
  const Tensor& std = this->var_.sqrt().expand_as(observations);
  return mean + observations * (std + EPS);
}

}  // namespace modules