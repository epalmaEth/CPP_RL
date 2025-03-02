#pragma once

#include <tensorboard_logger.h>

#include "algorithms/ppo.h"
#include "configs/configs.h"
#include "env/env.h"
#include "metrics.h"
#include "modules/actor_critic.h"
#include "storage/circular_buffer.h"
#include "storage/observation_buffer.h"
#include "utils/types.h"

namespace runners {

using TensorBoardLoggerPointer = std::unique_ptr<TensorBoardLogger>;

class OnPolicyRunner {
 public:
  OnPolicyRunner(const string& task, const configs::CfgPointer& cfg, const Device& device);

  void learn();
  void play();
  void save_models(const string& name) const;
  void load_models(const string& name, const bool& load_optimizer = false);
  const std::function<Tensor(const Tensor&)> get_inference_policy() const {
    return this->train_algorithm_->get_inference_policy();
  }

 private:
  void update_cfg_();
  void initialize_();
  void log_metric_(const TrainMetrics& metric) const;
  void train_() { this->train_algorithm_->train(); }
  void eval_() { this->train_algorithm_->eval(); }

  const configs::CfgPointer cfg_;
  env::EnvPointer env_;
  storage::ObservationBufferPointer observation_buffer_;
  algorithms::PPOPointer train_algorithm_;
  storage::CircularBufferFloatPointer reward_buffer_;
  storage::CircularBufferIntPointer length_buffer_;
  TensorBoardLoggerPointer logger_;

  const Device device_;
  env::Results env_results_;
  Tensor current_reward_sum_;
  Tensor current_episode_length_;
  float collection_time_ = 0.;
  float learn_time_ = 0.;
  float total_time_ = 0.;
  unsigned int total_time_steps_ = 0;
  unsigned int current_learning_iteration_ = 0;
};

using RunnerPointer = std::unique_ptr<OnPolicyRunner>;
}  // namespace runners