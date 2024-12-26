#pragma once

#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <vector>

#include "utils/types.h"
#include "env.h"

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

        float max_theta_init_ = M_PI;
        float max_theta_dot_init_ = 1.F;
        float max_theta_dot_ = 8.F;
        float max_action_ = 2.F;
        float dt_ = 0.05F;

        float g_ = 10.F;
        float m_ = 1.F;
        float l_ = 1.F;

        Tensor applied_torque_;
    };
} // namespace env


