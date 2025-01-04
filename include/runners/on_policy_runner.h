#pragma once

#include "algorithms/ppo.h"
#include "configs/configs.h"
#include "env/env.h"
#include "modules/actor_critic.h"
#include "utils/types.h"

namespace runners {

using Cfg = configs::Cfg;
using EnvPointer = std::shared_ptr<env::Env>;
using PPOPointer = std::unique_ptr<algorithms::PPO>;

struct TrainMetric {
  std::map<string, float> values;
  std::map<string, float> extra_values;
};

class OnPolicyRunner {
 public:
  OnPolicyRunner(const EnvPointer &env, Cfg &cfg, const Device &device);

  void learn();
  void save_models(const string &name) const;
  void load_models(const string &name, const bool &load_optimizer = false);
  std::function<Tensor(const Tensor &)> get_inference_policy() {
    return this->train_algorithm_->get_inference_policy();
  }

 private:
  void update_cfg_();
  void log_metrics_() const;
  void train_() { this->train_algorithm_->train(); };
  void eval_() { this->train_algorithm_->eval(); };

  Device device_;
  Cfg cfg_;
  EnvPointer env_ = nullptr;
  PPOPointer train_algorithm_ = nullptr;
  std::vector<TrainMetric> metrics_;

  unsigned int tot_time_steps_ = 0;
  unsigned int tot_time = 0;
  unsigned int current_learning_iteration_ = 0;
};

inline void print_metric(const std::string &name, const double value,
                         const std::string &unit = "") {
  const int pad = 35;  // Padding for alignment
  std::cout << std::left << std::setw(pad) << name << ":" << std::right
            << std::setw(pad) << value << unit << std::endl;
}

}  // namespace runners