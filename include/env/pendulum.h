#pragma once

#include "env.h"
#include "utils/types.h"

namespace env {
class PendulumEnv : public Env {
 public:
  // Inherit constructors from the Env class
  using Env::Env;

  ~PendulumEnv() override = default;

  void initialize_states() override;
  std::pair<Tensor, DictTensor> reset(const Tensor& indices) override;
  void close() override;
  int get_action_size() const override { return 1; }
  Tensor sample_action() const override;
  void update_render_data(DictListTensor& data) const override;
  void render(const DictListTensor& data) const override;

 private:
  void sample_state_(const Tensor& indices) override;
  void update_state_(const Tensor& action) override;
  void compute_observations_() override;
  void compute_reward_() override;
  void compute_terminated_() override;
  void compute_truncated_() override;
  void compute_info_() override;

  Tensor normalized_theta_() const;

  int max_iterations_ = 200;

  double max_theta_init_ = M_PI;
  double max_theta_dot_init_ = 1.;
  double max_theta_dot_ = 8.;
  double max_action_ = 2.;
  double dt_ = 0.05;

  double g_ = 10.;
  double m_ = 1.;
  double l_ = 1.;

  Tensor applied_torque_;
};
}  // namespace env
