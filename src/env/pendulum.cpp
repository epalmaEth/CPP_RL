#include "env/pendulum.h"

#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <iostream>

namespace env {

void PendulumEnv::initialize_states() {
  // Initialize the state
  this->state_ = torch::zeros({this->num_envs_, 2}).to(this->device_);
  this->applied_torque_ = torch::zeros({this->num_envs_}).to(this->device_);
  this->iteration_ = torch::zeros({this->num_envs_}).to(this->device_);
  this->step_result_ = {
      .observation = torch::zeros({this->num_envs_, 3}).to(this->device_),
      .reward = torch::zeros({this->num_envs_}).to(this->device_),
      .terminated =
          torch::zeros({this->num_envs_}).to(torch::kBool).to(this->device_),
      .truncated =
          torch::zeros({this->num_envs_}).to(torch::kBool).to(this->device_),
      .info = DictTensor()};
}

std::pair<Tensor, DictTensor> PendulumEnv::reset(const Tensor& indices) {
  if (indices.numel() > 0) {
    this->sample_state_(indices);
    this->iteration_.index_put_({indices}, 0);
    this->compute_observations_();
    this->compute_info_();
  }
  return std::make_pair(this->step_result_.observation,
                        this->step_result_.info);
}

void PendulumEnv::close() {
  // Close the environment
  return;
}

Tensor PendulumEnv::sample_action() const {
  return this->max_action_ *
         (2. * torch::rand({this->num_envs_, this->get_action_size()})
                   .to(this->device_) -
          1);
}

void PendulumEnv::update_render_data(DictListTensor& data) const {
  data["states"].push_back(this->state_.clone().to(torch::kCPU));
  data["actions"].push_back(this->applied_torque_.clone().to(torch::kCPU));
  data["rewards"].push_back(this->step_result_.reward.clone().to(torch::kCPU));
}

void PendulumEnv::render(const DictListTensor& data) const {
  // Create directories if they do not exist
  std::filesystem::create_directories("data/render/pendulum");
  // Open a file for writing the state values
  std::ofstream file("data/pendulum/render/data.csv");

  // Write a header with state, action, and reward columns
  file << "theta, theta_dot, action, reward" << std::endl;

  const ListTensor& states = data.at("states");
  const ListTensor& actions = data.at("actions");
  const ListTensor& rewards = data.at("rewards");

  // Export the states, actions, and rewards for each time step or iteration
  for (size_t i = 0; i < states.size(); ++i) {
    const Tensor& state = states[i];
    const Tensor& action = actions[i];
    const Tensor& reward = rewards[i];

    const float theta =
        state[0][0].item<float>();  // Assuming state is [theta, theta_dot]
    const float theta_dot = state[0][1].item<float>();
    const float action_value =
        action[0].item<float>();  // Assuming action is a scalar value
    const float reward_value =
        reward[0].item<float>();  // Assuming reward is a scalar value

    // Write state, action, and reward to the CSV file
    file << theta << "," << theta_dot << "," << action_value << ","
         << reward_value << std::endl;
  }

  // Close the file
  file.close();

  // Running python script
  string command = "python3 python/pendulum_plot.py";
  std::filesystem::create_directories("videos/pendulum");
  std::system(command.c_str());
}

void PendulumEnv::sample_state_(const Tensor& indices) {
  // Sample random states for each environment
  const Tensor& theta =
      this->max_theta_init_ *
      (2. * torch::rand({indices.numel()}).to(this->device_) - 1);
  const Tensor& theta_dot =
      this->max_theta_dot_init_ *
      (2. * torch::rand({indices.numel()}).to(this->device_) - 1);

  this->state_.index_put_({indices, 0}, theta);
  this->state_.index_put_({indices, 1}, theta_dot);
}

void PendulumEnv::update_state_(const Tensor& action) {
  Tensor theta = this->state_.index({this->all_indices_, 0});
  Tensor theta_dot = this->state_.index({this->all_indices_, 1});
  this->applied_torque_.index_put_(
      {this->all_indices_},
      torch::clamp(action.index({this->all_indices_, 0}), -this->max_action_,
                   this->max_action_));

  theta_dot += 3. * this->dt_ *
               (this->g_ / (2. * this->l_) * torch::sin(theta) +
                1. / (this->m_ * this->l_ * this->l_) * this->applied_torque_);
  theta_dot =
      torch::clamp(theta_dot, -this->max_theta_dot_, this->max_theta_dot_);
  theta += this->dt_ * theta_dot;

  this->state_.index_put_({this->all_indices_, 0}, theta);
  this->state_.index_put_({this->all_indices_, 1}, theta_dot);
}

void PendulumEnv::compute_observations_() {
  const Tensor& theta = this->state_.index({this->all_indices_, 0});
  const Tensor& theta_dot = this->state_.index({this->all_indices_, 1});
  this->step_result_.observation.index_put_({this->all_indices_, 0},
                                            torch::cos(theta));
  this->step_result_.observation.index_put_({this->all_indices_, 1},
                                            torch::sin(theta));
  this->step_result_.observation.index_put_({this->all_indices_, 2}, theta_dot);
}

void PendulumEnv::compute_reward_() {
  // Compute the reward
  const Tensor& theta_error = this->normalized_theta_().square();
  const Tensor& theta_dot_error =
      this->state_.index({this->all_indices_, 1}).square();
  const Tensor& torque_error = this->applied_torque_.square();
  this->step_result_.reward.index_put_(
      {this->all_indices_},
      -(theta_error + 0.1F * theta_dot_error + 0.001F * torque_error));
}

void PendulumEnv::compute_terminated_() { return; }

void PendulumEnv::compute_truncated_() {
  this->step_result_.truncated = this->iteration_ >= this->max_iterations_;
}

void PendulumEnv::compute_info_() { return; }

Tensor PendulumEnv::normalized_theta_() const {
  const Tensor& theta = this->state_.index({this->all_indices_, 0});
  return torch::remainder(theta + M_PI, 2. * M_PI) - M_PI;
}

}  // namespace env