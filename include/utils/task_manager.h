#pragma once

#include "env/env.h"
#include "env/pendulum.h"
#include "utils/types.h"

namespace utils {

using EnvPointer = std::shared_ptr<env::Env>;

class TaskManager {
 public:
  TaskManager() { this->register_basic_tasks_(); }

  void register_task(const string& task,
                     const std::function<EnvPointer(
                         const unsigned int&, const int&, const Device&)>& fn) {
    tasks_[task] = fn;
  }

  EnvPointer create_env(const string& task, const unsigned int& num_envs,
                        const int& seed, const Device& device) {
    if (tasks_.find(task) == tasks_.end())
      throw std::runtime_error("Task not found: " + task);
    return tasks_[task](num_envs, seed, device);
  }

 private:
  void register_basic_tasks_() {
    this->register_task(
        "pendulum", std::function<EnvPointer(const unsigned int&, const int&,
                                             const Device&)>(
                        [](const unsigned int& num_envs, const int& seed,
                           const Device& device) {
                          return std::make_shared<env::PendulumEnv>(
                              num_envs, seed, device);
                        }));
  }

  std::unordered_map<
      string,
      std::function<EnvPointer(const unsigned int&, const int&, const Device&)>>
      tasks_;
};
}  // namespace utils
