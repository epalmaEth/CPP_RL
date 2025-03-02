#include "env/physics_based_envs/pendulum_cart.h"

#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>

namespace env {

void PendulumCartEnv::initialize() {
  this->applied_force_ = torch::zeros({this->cfg_.num_envs}, this->device_);
  Env::initialize();
}

void PendulumCartEnv::close() {
  // Close the environment
  return;
}

const Tensor PendulumCartEnv::sample_action() const {
  return this->max_action_ *
         (2.f * torch::rand({this->cfg_.num_envs, this->get_action_size()}, this->device_) - 1.f);
}

void PendulumCartEnv::update_render_trajectory(const Results& results) const {
  std::ofstream file(
    "data/" + this->task_name_() + "/run_" + std::to_string(this->cfg_.run_id) + "/trajectory.csv",
    std::ios::app);

  float theta = this->state_[0][0].item<float>();
  float x = this->state_[0][1].item<float>();
  float action_value = this->applied_force_[0].item<float>();
  float reward_value = results.rewards[0].item<float>();

  // Write state, action, and rewards to the CSV file
  file << theta << "," << x << "," << action_value << "," << reward_value << std::endl;

  // Close the file
  file.close();
}

void PendulumCartEnv::render() const {
  // Running python script
  const string command = std::string("python3 python/pendulums_plot.py --use_cart ") +
                         "--task=pendulum_cart" + " --run_id=" + std::to_string(this->cfg_.run_id) +
                         " --rod_length=" + std::to_string(this->l_) + " --rod_width=0.01" +
                         " --bound=" + std::to_string(this->max_x_);
  if (std::system(command.c_str()) != 0)
    std::cout << "Error: Running python script " << command << std::endl;
}

void PendulumCartEnv::reset_state_(const int& num_resets, const Tensor& indices) {
  // Sample random states for each environment
  const Tensor theta =
    this->max_theta_init_ * (2.f * torch::rand({num_resets}, this->device_) - 1.f);
  const Tensor x = this->max_x_init_ / 2 * (2.f * torch::rand({num_resets}, this->device_) - 1.f);
  const Tensor theta_dot =
    this->max_theta_dot_init_ * (2.f * torch::rand({num_resets}, this->device_) - 1.f);
  const Tensor x_dot =
    this->max_x_dot_init_ * (2.f * torch::rand({num_resets}, this->device_) - 1.f);

  this->state_.index_put_({indices, 0}, theta);
  this->state_.index_put_({indices, 1}, x);
  this->state_.index_put_({indices, 2}, theta_dot);
  this->state_.index_put_({indices, 3}, x_dot);
}

void PendulumCartEnv::update_state_(const Tensor& action) {
  this->applied_force_.copy_(action.select(1, 0));
  this->applied_force_.clamp_(-this->max_action_, this->max_action_);

  if (this->cfg_.integrator == "euler")
    this->integrate_euler_(this->applied_force_);
  else if (this->cfg_.integrator == "rk2")
    this->integrate_rk2_(this->applied_force_);
  else if (this->cfg_.integrator == "rk4")
    this->integrate_rk4_(this->applied_force_);
  else
    throw std::invalid_argument("Invalid integrator: " + this->cfg_.integrator);
}

Tensor PendulumCartEnv::dynamics_(const Tensor& state, const Tensor& action) const {
  const Tensor theta = state.select(1, 0);
  const Tensor x = state.select(1, 1);
  const Tensor theta_dot = state.select(1, 2);
  const Tensor x_dot = state.select(1, 3);

  const Tensor sin_theta = torch::sin(theta);
  const Tensor cos_theta = torch::cos(theta);

  const Tensor theta_ddot =
    ((total_mass_ * this->g_ - factor_ * theta_dot.square() * cos_theta) * sin_theta -
     cos_theta * action) /
    ((2.f / 3.f * this->l_ * total_mass_ - factor_ * cos_theta.square()));

  const Tensor x_ddot =
    (action + factor_ * (theta_dot.square() * sin_theta - theta_ddot * cos_theta)) / total_mass_;

  return torch::stack({theta_dot, x_dot, theta_ddot, x_ddot}, 1);
}

void PendulumCartEnv::update_actor_obs_(Results& results) {
  const Tensor theta = this->state_.select(1, 0);
  const Tensor x = this->state_.select(1, 1);
  const Tensor theta_dot = this->state_.select(1, 2);
  const Tensor x_dot = this->state_.select(1, 3);

  results.actor_obs.select(1, 0).copy_(torch::cos(theta));
  results.actor_obs.select(1, 1).copy_(torch::sin(theta));
  results.actor_obs.select(1, 2).copy_(theta_dot);
  results.actor_obs.select(1, 3).copy_(x);
  results.actor_obs.select(1, 4).copy_(x_dot);
}

void PendulumCartEnv::update_critic_obs_(Results& results) {
  results.critic_obs.copy_(results.actor_obs);
}

void PendulumCartEnv::update_rewards_(Results& results) {
  const Tensor theta = this->state_.select(1, 0);
  const Tensor x_error = this->state_.select(1, 1).square();
  const Tensor force_error = this->applied_force_.square();
  results.rewards.copy_(torch::cos(theta) - 10.f * results.terminated.to(torch::kFloat) -
                        0.1f * x_error - 0.001f * force_error);
}

void PendulumCartEnv::update_terminated_(Results& results) {
  const Tensor x = this->state_.select(1, 1);
  results.terminated.copy_(x.abs() > this->max_x_);
}

const Tensor PendulumCartEnv::normalized_theta_() const {
  const Tensor theta = this->state_.select(1, 0);
  return torch::remainder(theta + M_PI, 2.f * M_PI) - M_PI;
}

void PendulumCartEnv::initialize_render_() {
  std::ofstream file("data/" + this->task_name_() + "/run_" + std::to_string(this->cfg_.run_id) +
                     "/trajectory.csv");

  // Write a header with state, action, and rewards columns
  file << "theta, x, action, rewards" << std::endl;

  // Close the file
  file.close();
}

}  // namespace env