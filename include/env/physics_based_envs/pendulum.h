#pragma once

#include "configs/configs.h"
#include "physics_based_env.h"
#include "utils/types.h"

namespace env {
class PendulumEnv : public PhysicsBasedEnv {
 public:
  // Inherit the constructor from the base class
  using PhysicsBasedEnv::PhysicsBasedEnv;

  ~PendulumEnv() override = default;

  void initialize() override;
  void close() override;
  unsigned int get_actor_obs_size() const override { return 3; }
  unsigned int get_action_size() const override { return 1; }
  const Tensor get_action_min() const override {
    return torch::full((this->get_action_size()), -this->max_action_, this->device_);
  }
  const Tensor get_action_max() const override {
    return torch::full((this->get_action_size()), this->max_action_, this->device_);
  }
  const Tensor sample_action() const override;
  void update_render_trajectory(const Results& results) const override;
  void render() const override;

 private:
  unsigned int get_state_size_() const override { return 2; }
  void reset_state_(const int& num_resets, const Tensor& indices) override;
  void update_state_(const Tensor& action) override;
  Tensor dynamics_(const Tensor& state, const Tensor& action) const override;

  void update_actor_obs_(Results& results) override;
  void update_critic_obs_(Results& results) override;
  void update_rewards_(Results& results) override;

  void initialize_render_() override;
  string task_name_() const override { return "pendulum"; }

  const Tensor normalized_theta_() const;

  const float max_theta_init_ = M_PI;
  const float max_theta_dot_init_ = 1.f;
  const float max_theta_dot_ = 8.f;
  const float max_action_ = 2.f;

  const float g_ = 10.f;
  const float m_ = 1.f;
  const float l_ = 1.f;

  Tensor applied_torque_;
};
}  // namespace env
