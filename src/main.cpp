#include <torch/torch.h>
#include <torch/cuda.h>
#include <iostream>

#include "env/env.h"
#include "env/pendulum.h"
#include "utils/task_manager.h"
#include "utils/types.h"

int main() {
    
    utils::register_eval_tasks();

    std::string task = "pendulum";
    std::cout << "Task: " << task << std::endl;

    const Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Device: " << device << "\n";

    std::cout << "-------Creating environment-------" << std::endl;
    const int seed = 0;
    std::unique_ptr<env::Env> env = utils::TaskManager::create_env(task, device, seed);

    if (env == nullptr) {
        std::cerr << "[Error] Task not supported" << std::endl;
        return 0;
    }
    
    env->initialize_states();
    auto [obs, info] = env->reset(env->get_all_indices());
    
    std::cout << "-------Starting simulation-------" << std::endl;
    DictListTensor data;
    while (true) {
        auto action = env->sample_action();
        auto step_result = env->step(action);
        env->update_render_data(data);

        if ((step_result.terminated | step_result.truncated).all().item<bool>()) {
            break;
        }
    }
    std::cout << "-------Ending Simulation-------" << std::endl;
    env->close();

    std::cout << "-------Exporting states-------" << std::endl;
    env->export_data(data);
    
    return 0;
}