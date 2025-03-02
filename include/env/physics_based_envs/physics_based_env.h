#pragma once

#include <torch/torch.h>

#include "env/env.h"
#include "utils/types.h"

namespace env {

class PhysicsBasedEnv : public Env {
 public:
  // Inherit the constructor from the base class
  PhysicsBasedEnv(const configs::EnvCfg& cfg, const Device& device)
    : Env(cfg, device), dt_(cfg.dt) {}

  virtual ~PhysicsBasedEnv() override = default;

 protected:
  virtual Tensor dynamics_(const Tensor& state, const Tensor& action) const = 0;

  void integrate_euler_(const Tensor& action) {
    this->state_.add_(this->dt_ * this->dynamics_(this->state_, action));
  }

  void integrate_rk2_(const Tensor& action) {
    const Tensor k1 = this->dynamics_(this->state_, action);
    const Tensor k2 = this->dynamics_(this->state_ + this->dt_ * k1, action);
    this->state_.add_(0.5 * this->dt_ * (k1 + k2));
  }

  void integrate_rk4_(const Tensor& action) {
    const Tensor k1 = this->dynamics_(this->state_, action);
    const Tensor k2 = this->dynamics_(this->state_ + 0.5 * this->dt_ * k1, action);
    const Tensor k3 = this->dynamics_(this->state_ + 0.5 * this->dt_ * k2, action);
    const Tensor k4 = this->dynamics_(this->state_ + this->dt_ * k3, action);
    this->state_.add_(this->dt_ / 6. * (k1 + 2. * k2 + 2. * k3 + k4));
  }

  float dt_;
};

}  // namespace env