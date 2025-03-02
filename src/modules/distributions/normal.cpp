#include "modules/distributions/normal.h"

namespace modules {

const Tensor Normal::get_kl(const DictTensor& old_kl_params) const {
  const Tensor& old_mean = old_kl_params.at("mean");
  const Tensor& old_std = old_kl_params.at("std");
  const Tensor& new_std = this->std_;
  const Tensor& new_mean = this->mean_;

  return torch::log(new_std / old_std) +
         (old_std.square() + (old_mean - new_mean).square()) / (2. * new_std.square()) - 0.5;
}

}  // namespace modules