#include "env/pendulum.h"
#include <iostream>

namespace env {

    void PendulumEnv::initialize_states() {
        // Initialize the state
        this->state_ = torch::zeros({this->num_envs_, 2}).to(this->device_);
        this->iteration_ = torch::zeros({this->num_envs_}).to(this->device_);
        this->step_result_ = {
            .observation = torch::zeros({this->num_envs_, 3}).to(this->device_),
            .reward = torch::zeros({this->num_envs_}).to(this->device_),
            .terminated = torch::zeros({this->num_envs_}).to(torch::kBool).to(this->device_),
            .truncated = torch::zeros({this->num_envs_}).to(torch::kBool).to(this->device_),
            .info = TensorDict()
        };
    }

    std::pair<Tensor, TensorDict> PendulumEnv::reset(const Tensor& indices) {
        if(indices.numel() > 0) {
            this->sample_state_(indices);
            this->iteration_.index_put_({indices}, 0);
            this->compute_observations_();
            this->compute_info_();
        }
        return std::make_pair(this->step_result_.observation, this->step_result_.info);
    }

    void PendulumEnv::render() const {
        const float theta = this->state_[0][0].item<float>();
        const float theta_dot = this->state_[0][1].item<float>();

        std::cout << "theta: " << theta << ", theta_dot: " << theta_dot << std::endl;
    }

    void PendulumEnv::close() {
        // Close the environment
        std::cout << "Closing the pendulum environment" << std::endl;
    }

    Tensor PendulumEnv::sample_action() {
        return this->max_action_*(2*torch::rand({this->num_envs_, this->get_action_size()}).to(this->device_) - 1);
    }

    void PendulumEnv::sample_state_(const Tensor& indices) {
        // Sample random states for each environment
        const Tensor& theta = this->max_theta_init_*(2*torch::rand({indices.numel()}).to(this->device_) - 1);  
        const Tensor& theta_dot = this->max_theta_dot_init_*(2*torch::rand({indices.numel()}).to(this->device_) - 1);  

        this->state_.index_put_({indices, 0}, theta);
        this->state_.index_put_({indices, 1}, theta_dot);
    }

    void PendulumEnv::update_state_(const Tensor& action) {
        Tensor theta = this->state_.index({this->all_indices_, 0});
        Tensor theta_dot = this->state_.index({this->all_indices_, 1});
        const Tensor& torque = torch::clamp(action.index({this->all_indices_, 0}), -this->max_action_, this->max_action_);

        theta_dot += 3.F*this->dt_*(this->g_/(2.F*this->l_)*torch::sin(theta) + 1.F/(this->m_*this->l_*this->l_)*torque);
        theta_dot = torch::clamp(theta_dot, -this->max_theta_dot_, this->max_theta_dot_);
        theta += this->dt_*theta_dot;

        this->state_.index_put_({this->all_indices_, 0}, theta);
        this->state_.index_put_({this->all_indices_, 1}, theta_dot);
    }

    void PendulumEnv::compute_observations_() {
        const Tensor& theta = this->state_.index({this->all_indices_, 0});
        const Tensor& theta_dot = this->state_.index({this->all_indices_, 1});
        this->step_result_.observation.index_put_({this->all_indices_, 0}, torch::cos(theta));
        this->step_result_.observation.index_put_({this->all_indices_, 1}, torch::sin(theta));
        this->step_result_.observation.index_put_({this->all_indices_, 2}, theta_dot);
    }

    void PendulumEnv::compute_reward_(const Tensor& action) {
        // Compute the reward
        const Tensor& theta_error = this->normalized_theta_().pow(2);
        const Tensor& theta_dot_error = this->state_.index({this->all_indices_, 1}).pow(2);
        const Tensor& torque_error = torch::clamp(action.index({this->all_indices_, 0}), -this->max_action_, this->max_action_).pow(2);
        this->step_result_.reward.index_put_({this->all_indices_}, -(theta_error + 0.1F*theta_dot_error + 0.001F*torque_error));
    }

    void PendulumEnv::compute_terminated_() {
        return;
    }
    
    void PendulumEnv::compute_truncated_() {
        this->step_result_.truncated = this->iteration_ >= this->max_iterations_;
    }


    void PendulumEnv::compute_info_() {
        return;
    }

    Tensor PendulumEnv::normalized_theta_() const {
        const Tensor& theta = this->state_.index({this->all_indices_, 0});
        return torch::remainder(theta + M_PI, 2*M_PI) - M_PI;
    }

} // namespace env