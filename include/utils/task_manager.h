#pragma once

#include "env/env.h"
#include "env/pendulum.h"
#include "utils/types.h"

namespace utils {
class TaskManager {
 public:
  // Task map to hold task registrations
  template <typename... Args>
  static std::map<string, std::function<std::unique_ptr<env::Env>(Args...)>>
      task_map;

  // Register task function
  template <typename... Args>
  static void register_task(
      const string& task,
      std::function<std::unique_ptr<env::Env>(Args...)> fn) {
    task_map<Args...>[task] = fn;
  }

  // Create environment function
  template <typename... Args>
  static std::unique_ptr<env::Env> create_env(const string& task,
                                              Args&&... args) {
    auto it = task_map<Args...>.find(task);
    if (it != task_map<Args...>.end()) {
      return it->second(std::forward<Args>(args)...);
    }
    return nullptr;
  }
};

// Definition of the static task_map outside the class template
template <typename... Args>
std::map<string, std::function<std::unique_ptr<env::Env>(Args...)>>
    TaskManager::task_map;

void register_eval_tasks() {
  // Register tasks
  TaskManager::register_task<const Device&, const int&>(
      "pendulum",
      std::function<std::unique_ptr<env::Env>(const Device&, const int&)>(
          [](const Device& device, const int& seed) {
            return std::make_unique<env::PendulumEnv>(device, seed);
          }));
}

}  // namespace utils
