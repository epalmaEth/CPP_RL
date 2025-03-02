#pragma once

#include "algorithms/ppo.h"
#include "utils/types.h"
#include "utils/utils.h"

namespace runners {

struct ComputationMetrics {
  const float fps;
  const float iteration_time;
  const float collection_time;
  const float learn_time;

  ComputationMetrics(const float& fps, const float& iteration_time, const float& collection_time,
                     const float& learn_time)
    : fps(fps),
      iteration_time(iteration_time),
      collection_time(collection_time),
      learn_time(learn_time) {}

  friend std::ostream& operator<<(std::ostream& os, const ComputationMetrics& metrics) {
    os << "\033[1mComputation Metrics:\033[0m" << std::endl;
    os << utils::formatOutput("Fps", metrics.fps, " Step/s") << std::endl;
    os << utils::formatOutput("Iteration Time", metrics.iteration_time, " s") << std::endl;
    os << utils::formatOutput("Collection Time", metrics.collection_time, " s") << std::endl;
    os << utils::formatOutput("Learning Time", metrics.learn_time, " s") << std::endl;
    return os;
  }
};

struct RewardMetrics {
  const float reward;
  const float length;

  RewardMetrics(const float& reward, const float& length) : reward(reward), length(length) {}

  friend std::ostream& operator<<(std::ostream& os, const RewardMetrics& metrics) {
    os << "\033[1mReward Metrics:\033[0m" << std::endl;
    os << utils::formatOutput("Mean Reward", metrics.reward) << std::endl;
    os << utils::formatOutput("Mean Length", metrics.length) << std::endl;
    return os;
  }
};

struct TotalMetrics {
  const unsigned int total_time_steps;
  const float total_time;

  TotalMetrics(const unsigned int& total_time_steps, const float& total_time)
    : total_time_steps(total_time_steps), total_time(total_time) {}

  friend std::ostream& operator<<(std::ostream& os, const TotalMetrics& metrics) {
    os << "\033[1mTotal Metrics:\033[0m" << std::endl;
    os << utils::formatOutput("Total Timesteps", metrics.total_time_steps) << std::endl;
    os << utils::formatOutput("Total Time", metrics.total_time, " s") << std::endl;
    return os;
  }
};

struct ExtraMetrics {
  const std::map<string, float> values;

  explicit ExtraMetrics(const std::map<string, float>& values) : values(values) {}

  friend std::ostream& operator<<(std::ostream& os, const ExtraMetrics& metrics) {
    if (metrics.values.empty()) return os;
    os << "\033[1mExtra Metrics:\033[0m" << std::endl;
    for (const auto& [key, value] : metrics.values) {
      os << utils::formatOutput(key, value) << std::endl;
    }
    return os;
  }
};

struct TrainMetrics {
  const unsigned int current_iteration;
  const unsigned int end_iteration;
  const ComputationMetrics computation_metrics;
  const algorithms::LossMetrics loss_metrics;
  const RewardMetrics reward_metrics;
  const TotalMetrics total_metrics;
  const ExtraMetrics extra_metrics;

  TrainMetrics(const unsigned int& current_iteration, const unsigned int& end_iteration,
               const ComputationMetrics& computation_metrics,
               const algorithms::LossMetrics& loss_metrics, const RewardMetrics& reward_metrics,
               const TotalMetrics& total_metrics, const ExtraMetrics& extra_metrics)
    : current_iteration(current_iteration),
      end_iteration(end_iteration),
      computation_metrics(computation_metrics),
      loss_metrics(loss_metrics),
      reward_metrics(reward_metrics),
      total_metrics(total_metrics),
      extra_metrics(extra_metrics) {}

  const std::map<string, float> to_dict() const {
    std::map<string, float> dict;
    dict["Perf/fps"] = computation_metrics.fps;
    dict["Perf/collection_time"] = computation_metrics.collection_time;
    dict["Perf/learn_time"] = computation_metrics.learn_time;
    dict["Loss/actor_loss"] = loss_metrics.actor_loss;
    dict["Loss/critic_loss"] = loss_metrics.critic_loss;
    dict["Loss/entropy_loss"] = loss_metrics.entropy_loss;
    dict["Loss/kl_loss"] = loss_metrics.kl_loss;
    dict["Train/reward"] = reward_metrics.reward;
    dict["Train/length"] = reward_metrics.length;
    for (const auto& [key, value] : extra_metrics.values) dict["Extra/" + key] = value;
    return dict;
  }

  friend std::ostream& operator<<(std::ostream& os, const TrainMetrics& metric) {
    os << std::endl << string(metric.width_, '#') << std::endl;
    os << "\033[1m Learning Iteration " << metric.current_iteration << " / " << metric.end_iteration
       << " \033[0m" << std::endl;
    os << string(metric.width_, '#') << std::endl << std::endl;
    os << metric.computation_metrics << std::endl;
    os << metric.loss_metrics << std::endl;
    os << metric.reward_metrics << std::endl;
    os << metric.extra_metrics << std::endl;
    os << metric.total_metrics << std::endl;
    float eta = metric.total_metrics.total_time / (metric.current_iteration) *
                (metric.end_iteration - metric.current_iteration + 1);
    os << utils::formatOutput("ETA", eta, " s") << std::endl;
    return os;
  }

 private:
  const unsigned int width_ = 80;  // Width for separators
};

}  // namespace runners