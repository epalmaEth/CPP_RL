#include "modules/distributions/beta.h"

#include <torch/torch.h>

namespace modules {

void Beta::update(const Tensor& hidden_output) {
  this->mean_ = hidden_output.clamp(this->min_, this->max_);

  const Tensor& var = this->std_.expand_as(this->mean_).square();

  const Tensor& total = (this->min_ * this->max_ - (this->min_ + this->max_) * this->mean_ +
                         this->mean_.square() + var) /
                        var / (this->max_ - this->min_);
  this->alpha_ = (this->min_ - this->mean_) * total;
  this->beta_ = (this->mean_ - this->max_) * total;
}

const Tensor Beta::get_log_prob(const Tensor& actions) const {
  const Tensor& unscaled_actions = this->unscale_(actions);
  return (this->alpha_ - 1.f) * torch::log(unscaled_actions) +
         (this->beta_ - 1.f) * torch::log(1.f - unscaled_actions) -
         (torch::lgamma(this->alpha_) + torch::lgamma(this->beta_) -
          torch::lgamma(this->alpha_ + this->beta_)) -
         torch::log(this->max_ - this->min_);
}

const Tensor Beta::get_entropy() const {
  return torch::lgamma(this->alpha_) + torch::lgamma(this->beta_) -
         torch::lgamma(this->alpha_ + this->beta_) -
         (this->alpha_ - 1.f) *
           (torch::digamma(this->alpha_) - torch::digamma(this->alpha_ + this->beta_)) -
         (this->beta_ - 1.f) *
           (torch::digamma(this->beta_) - torch::digamma(this->alpha_ + this->beta_)) +
         torch::log(this->max_ - this->min_);
}

const Tensor Beta::get_kl(const DictTensor& old_kl_params) const {
  const Tensor& alpha = this->alpha_;
  const Tensor& beta = this->beta_;
  const Tensor& old_alpha = old_kl_params.at("alpha");
  const Tensor& old_beta = old_kl_params.at("beta");

  return torch::lgamma(alpha) + torch::lgamma(beta) - torch::lgamma(alpha + beta) -
         torch::lgamma(old_alpha) - torch::lgamma(old_beta) + torch::lgamma(old_alpha + old_beta) +
         (old_alpha - alpha) * (torch::digamma(old_alpha) - torch::digamma(old_alpha + old_beta)) +
         (old_beta - beta) * (torch::digamma(old_beta) - torch::digamma(old_alpha + old_beta)) -
         torch::log(this->max_ - this->min_);
}

const DictTensor Beta::get_kl_params() const {
  return {{"alpha", this->alpha_}, {"beta", this->beta_}};
}

}  // namespace modules