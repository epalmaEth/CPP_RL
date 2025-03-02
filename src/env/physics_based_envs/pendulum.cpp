#include "env/physics_based_envs/pendulum.h"

#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>

namespace env {

void PendulumEnv::initialize() {
  this->applied_torque_ = torch::zeros({this->cfg_.num_envs}, this->device_);
  Env::initialize();
}

void PendulumEnv::close() {
  // Close the environment
  return;
}

const Tensor PendulumEnv::sample_action() const {
  return this->max_action_ *
         (2.f * torch::rand({this->cfg_.num_envs, this->get_action_size()}, this->device_) - 1.f);
}

void PendulumEnv::update_render_trajectory(const Results& results) const {
  std::ofstream file(
    "data/" + this->task_name_() + "/run_" + std::to_string(this->cfg_.run_id) + "/trajectory.csv",
    std::ios::app);

  float theta = this->state_[0][0].item<float>();
  float action_value = this->applied_torque_[0].item<float>();
  float reward_value = results.rewards[0].item<float>();

  // Write state, action, and rewards to the CSV file
  file << theta << "," << action_value << "," << reward_value << std::endl;

  // Close the file
  file.close();
}

void PendulumEnv::render() const {
  // Running python script
  const string command = std::string("python3 python/pendulums_plot.py --task=pendulum") +
                         " --run_id=" + std::to_string(this->cfg_.run_id) +
                         " --rod_length=" + std::to_string(this->l_);

  if (std::system(command.c_str()) != 0)
    std::cout << "Error: Running python script " << command << std::endl;
}

void PendulumEnv::reset_state_(const int& num_resets, const Tensor& indices) {
  // Sample random states for each environment
  const Tensor theta =
    this->max_theta_init_ * (2.f * torch::rand({num_resets}, this->device_) - 1.f);
  const Tensor theta_dot =
    this->max_theta_dot_init_ * (2.f * torch::rand({num_resets}, this->device_) - 1.f);

  this->state_.index_put_({indices, 0}, theta);
  this->state_.index_put_({indices, 1}, theta_dot);
}

void PendulumEnv::update_state_(const Tensor& action) {
  this->applied_torque_.copy_(action.select(1, 0));
  this->applied_torque_.clamp_(-this->max_action_, this->max_action_);

  if (this->cfg_.integrator == "euler")
    this->integrate_euler_(this->applied_torque_);
  else if (this->cfg_.integrator == "rk2")
    this->integrate_rk2_(this->applied_torque_);
  else if (this->cfg_.integrator == "rk4")
    this->integrate_rk4_(this->applied_torque_);
  else
    throw std::invalid_argument("Invalid integrator: " + this->cfg_.integrator);

  this->state_.select(1, 1).clamp_(-this->max_theta_dot_, this->max_theta_dot_);
}

Tensor PendulumEnv::dynamics_(const Tensor& state, const Tensor& action) const {
  const Tensor theta = state.select(1, 0);
  const Tensor theta_dot = state.select(1, 1);

  Tensor theta_ddot =
    3.f / this->l_ * (this->g_ / 2.f * torch::sin(theta) + 1.f / (this->m_ * this->l_) * action);

  return torch::stack({theta_dot, theta_ddot}, 1);
}

void PendulumEnv::update_actor_obs_(Results& results) {
  const Tensor theta = this->state_.select(1, 0);
  const Tensor theta_dot = this->state_.select(1, 1);

  results.actor_obs.select(1, 0).copy_(torch::cos(theta));
  results.actor_obs.select(1, 1).copy_(torch::sin(theta));
  results.actor_obs.select(1, 2).copy_(theta_dot);
}

void PendulumEnv::update_critic_obs_(Results& results) {
  results.critic_obs.copy_(results.actor_obs);
}

void PendulumEnv::update_rewards_(Results& results) {
  const Tensor theta_error = this->normalized_theta_().square();
  const Tensor theta_dot_error = this->state_.select(1, 1).square();
  const Tensor torque_error = this->applied_torque_.square();
  results.rewards.copy_(-(theta_error + 0.1f * theta_dot_error + 0.001f * torque_error));
}

const Tensor PendulumEnv::normalized_theta_() const {
  const Tensor theta = this->state_.select(1, 0);
  return torch::remainder(theta + M_PI, 2.f * M_PI) - M_PI;
}

void PendulumEnv::initialize_render_() {
  std::ofstream file("data/" + this->task_name_() + "/run_" + std::to_string(this->cfg_.run_id) +
                     "/trajectory.csv");

  // Write a header with state, action, and rewards columns
  file << "theta, action, rewards" << std::endl;

  // Close the file
  file.close();
}

}  // namespace env