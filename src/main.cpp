#include <torch/torch.h>
#include <torch/cuda.h>
#include <iostream>

#include "env/pendulum.h"
#include "utils/types.h"
#include "utils/export.h"

int main() {

    std::cout << "Checking device" << std::endl;
    const Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Device: " << device << "\n";

    std::cout << "Creating environment" << std::endl;
    const int num_envs = 1;
    env::PendulumEnv env(num_envs, device);
    env.initialize_states();
    auto [obs, info] = env.reset(env.get_all_indices());
    
    std::cout << "Starting simulation" << std::endl;
    std::vector<Tensor> states;
    std::vector<Tensor> actions;
    std::vector<Tensor> rewards;
    while (true) {
        auto action = env.sample_action();
        auto step_result = env.step(action);

        states.push_back(env.get_state().clone());
        actions.push_back(action.clone());
        rewards.push_back(step_result.reward.clone());

        if ((step_result.terminated | step_result.truncated).all().item<bool>()) {
            break;
        }
    }
    std::cout << "Simulation done" << std::endl;

    std::cout << "Exporting states" << std::endl;
    utils::export_states(states, actions, rewards);
    return 0;
}