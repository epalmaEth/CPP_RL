#include <iostream>
#include <fstream>
#include <cmath>
#include <torch/torch.h>

#include "utils/types.h"

namespace utils {

    void export_states(const std::vector<torch::Tensor>& states, 
                    const std::vector<torch::Tensor>& actions, 
                    const std::vector<torch::Tensor>& rewards) {

        // Open a file for writing the state values
        std::ofstream file("data/pendulum/data.csv");
        
        // Write a header with state, action, and reward columns
        file << "theta, theta_dot, action, reward" << std::endl;

        // Export the states, actions, and rewards for each time step or iteration
        for (size_t i = 0; i < states.size(); ++i) {
            const torch::Tensor& state = states[i];
            const torch::Tensor& action = actions[i];
            const torch::Tensor& reward = rewards[i];

            const float theta = state[0][0].item<float>();  // Assuming state is [theta, theta_dot]
            const float theta_dot = state[0][1].item<float>();
            const float action_value = action[0][0].item<float>();  // Assuming action is a scalar value
            const float reward_value = reward[0].item<float>();  // Assuming reward is a scalar value
            
            // Write state, action, and reward to the CSV file
            file << theta << "," << theta_dot << "," << action_value << "," << reward_value << std::endl;
        }

        // Close the file
        file.close();
    }

} // namespace utils
