#include "runners/on_policy_runner.h"

#include <torch/torch.h>

#include "utils/circular_buffer.h"
#include "utils/task_manager.h"

namespace runners {

OnPolicyRunner::OnPolicyRunner(const EnvPointer &env, Cfg &cfg,
                               const Device &device)
    : env_(env), cfg_(cfg), device_(device) {
  this->update_cfg_();
  std::cout << this->cfg_ << std::endl;
  this->train_algorithm_ =
      std::make_unique<algorithms::PPO>(this->cfg_, device);
  std::cout << "OnPolicyRunner created" << std::endl;
}

void OnPolicyRunner::learn() {
  int total_time_steps = 0;

  float collection_time = 0.;
  float learn_time = 0.;
  float total_time = 0.;

  Tensor current_reward_sum =
      torch::zeros({this->cfg_.runner_cfg.num_envs}).to(this->device_);
  Tensor current_episode_length = torch::zeros({this->cfg_.runner_cfg.num_envs})
                                      .to(torch::kInt32)
                                      .to(this->device_);

  utils::CircularBufferFloat reward_buffer =
      utils::CircularBufferFloat(this->cfg_.runner_cfg.logging_buffer);
  utils::CircularBufferInt length_buffer =
      utils::CircularBufferInt(this->cfg_.runner_cfg.logging_buffer);

  this->train_();
  auto [actor_obs, info] = this->env_->reset(this->env_->get_all_indices());
  Tensor critic_obs = info.count("critic") ? info.at("critic") : actor_obs;

  const int start = this->current_learning_iteration_;
  const int end = this->cfg_.runner_cfg.max_iterations + start;

  for (int it = start; it < end; ++it) {
    auto start = std::chrono::high_resolution_clock::now();

    // Rollout
    {
      torch::NoGradGuard no_grad;
      for (int i = 0; i < this->cfg_.runner_cfg.num_steps_per_env; ++i) {
        const Tensor &actions =
            this->train_algorithm_->act(actor_obs, critic_obs);
        const StepResult &step_return = this->env_->step(actions);
        this->train_algorithm_->process_step(
            step_return.reward, step_return.terminated, step_return.truncated);

        current_reward_sum += step_return.reward;
        current_episode_length += 1;

        const Tensor &done_ids =
            (step_return.terminated | step_return.truncated);

        if (done_ids.sum().item<int>() > 0) {
          reward_buffer.push(
              current_reward_sum.index({done_ids}).cpu().mean().item<float>());
          length_buffer.push(current_episode_length.index({done_ids})
                                 .cpu()
                                 .to(torch::kFloat32)
                                 .mean()
                                 .item<float>());

          current_reward_sum.index_put_({done_ids}, 0.0);
          current_episode_length.index_put_({done_ids}, 0);

          const auto &result = this->env_->reset(done_ids);
          actor_obs.index_put_({Slice(), Slice()}, result.first);
          critic_obs.index_put_({Slice(), Slice()},
                                result.second.count("critic")
                                    ? result.second.at("critic")
                                    : actor_obs);
        }
      }
      collection_time = std::chrono::duration<float>(
                            std::chrono::high_resolution_clock::now() - start)
                            .count();

      // Learning step
      start = std::chrono::high_resolution_clock::now();
      this->train_algorithm_->compute_returns(critic_obs);
    }
    const auto &[mean_value_loss, mean_surrogate_loss] =
        this->train_algorithm_->update_actor_critic();
    learn_time = std::chrono::duration<float>(
                     std::chrono::high_resolution_clock::now() - start)
                     .count();
    this->current_learning_iteration_ = it;

    // Logging
    TrainMetric metric;
    float time = collection_time + learn_time;
    int time_steps = this->cfg_.runner_cfg.num_envs *
                     this->cfg_.runner_cfg.num_steps_per_env;
    total_time += time;
    total_time_steps += time_steps;
    metric.values["end"] = end;
    metric.values["fps"] = time_steps / time;
    metric.values["collection_time"] = collection_time;
    metric.values["learn_time"] = learn_time;
    metric.values["mean_value_loss"] = mean_value_loss.item<float>();
    metric.values["mean_surrogate_loss"] = mean_surrogate_loss.item<float>();
    const Tensor &mean_stds = this->train_algorithm_->get_action_std().cpu();
    for (int i = 0; i < mean_stds.size(0); ++i) {
      metric.extra_values["action_std_" + std::to_string(i)] =
          mean_stds[i].item<float>();
    }
    if (reward_buffer.size() > 0) {
      metric.extra_values["mean_reward"] = reward_buffer.mean();
      metric.extra_values["mean_length"] = length_buffer.mean();
    }
    metric.values["tot_time_steps"] = total_time_steps;
    metric.values["it_time"] = time;
    metric.values["tot_time"] = total_time;
    this->metrics_.push_back(metric);
    this->log_metrics_();

    // Save models
    if (it + 1 % this->cfg_.runner_cfg.save_interval == 0)
      this->save_models("/models_" +
                        std::to_string(this->current_learning_iteration_ + 1) +
                        ".pt");
  }
  // Save models
  this->save_models("/models_last.pt");
}

void OnPolicyRunner::save_models(const string &name) const {
  torch::serialize::OutputArchive archive =
      this->train_algorithm_->save_models();
  archive.save_to(this->cfg_.runner_cfg.save_dir + name);
}

void OnPolicyRunner::load_models(const string &name,
                                 const bool &load_optimizer) {
  torch::serialize::InputArchive archive;
  archive.load_from(this->cfg_.runner_cfg.load_dir + name);
  this->train_algorithm_->load_models(archive, load_optimizer);
}

void OnPolicyRunner::update_cfg_() {
  const auto &[obs, info] = this->env_->reset(this->env_->get_all_indices());
  const unsigned int &num_actor_obs = obs.size(1);
  unsigned int num_critic_obs =
      info.count("critic") ? info.at("critic").size(1) : num_actor_obs;
  this->cfg_.update(num_actor_obs, num_critic_obs,
                    this->env_->get_action_size());
}

void OnPolicyRunner::log_metrics_() const {
  const int width = 80;  // Width for separators
  const int pad = 35;    // Padding for alignment
  const TrainMetric &metric = this->metrics_.back();

  // Separator and iteration header
  std::cout << "\n" << std::string(width, '#') << std::endl;
  std::cout << "\033[1m Learning Iteration "
            << this->current_learning_iteration_ + 1 << " / "
            << metric.values.at("end") << " \033[0m" << std::endl;
  std::cout << std::string(width, '#') << "\n" << std::endl;

  // Computation metrics
  std::cout << "\033[1mComputation Metrics:\033[0m" << std::endl;
  print_metric("Steps/s", metric.values.at("fps"));
  print_metric("Iteration Time", metric.values.at("it_time"), " s");
  print_metric("Collection Time", metric.values.at("collection_time"), " s");
  print_metric("Learning Time", metric.values.at("learn_time"), " s");
  std::cout << std::endl;

  // Loss metrics
  std::cout << "\033[1mLoss Metrics:\033[0m" << std::endl;
  print_metric("Value Function Loss", metric.values.at("mean_value_loss"));
  print_metric("Surrogate Loss", metric.values.at("mean_surrogate_loss"));
  std::cout << std::endl;

  // Extra metrics
  if (!metric.extra_values.empty()) {
    std::cout << "\033[1mExtra Metrics:\033[0m" << std::endl;
    for (const auto &extra : metric.extra_values) {
      print_metric(extra.first, extra.second);
    }
    std::cout << std::endl;
  }

  // Timing metrics
  std::cout << "\033[1mTotal Metrics:\033[0m" << std::endl;
  print_metric("Total Timesteps", metric.values.at("tot_time_steps"));
  print_metric("Total Time", metric.values.at("tot_time"), " s");

  // ETA
  float eta = metric.values.at("tot_time") /
              (this->current_learning_iteration_ + 1) *
              (metric.values.at("end") - this->current_learning_iteration_);
  print_metric("ETA", eta, " s");
}

}  // namespace runners