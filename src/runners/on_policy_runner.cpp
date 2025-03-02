#include "runners/on_policy_runner.h"

#include <torch/torch.h>

#include "env/task_manager.h"
#include "utils/utils.h"

namespace runners {

OnPolicyRunner::OnPolicyRunner(const string& task, const configs::CfgPointer& cfg,
                               const Device& device)
  : cfg_(cfg), device_(device) {
  this->env_ = std::move(env::TaskManager::create(task, cfg->env_cfg, device));
  this->observation_buffer_ = std::make_unique<storage::ObservationBuffer>(
    cfg, this->env_->get_actor_obs_size(), this->env_->get_critic_obs_size(),
    this->env_->get_action_size(), device);
  this->update_cfg_();
  std::cout << *this->cfg_ << std::endl;
  this->train_algorithm_ = std::make_unique<algorithms::PPO>(cfg, device);
  this->reward_buffer_ =
    std::make_unique<storage::CircularBufferFloat>(cfg->runner_cfg.logging_buffer);
  this->length_buffer_ =
    std::make_unique<storage::CircularBufferInt>(cfg->runner_cfg.logging_buffer);
  const string run_path = utils::get_run_path(this->cfg_->env_cfg.task);
  this->logger_ = std::make_unique<TensorBoardLogger>(run_path + "/tensorboard.tfevents");

  this->initialize_();
}

void OnPolicyRunner::learn() {
  this->total_time_steps_ = 0;

  this->collection_time_ = 0.;
  this->learn_time_ = 0.;
  this->total_time_ = 0.;

  this->current_reward_sum_.zero_();
  this->current_episode_length_.zero_();

  this->reward_buffer_->clear();
  this->length_buffer_->clear();

  this->train_();

  this->env_->reset(this->env_results_);
  this->observation_buffer_->reset(this->env_results_);

  unsigned int start = this->current_learning_iteration_;
  unsigned int end = this->cfg_->runner_cfg.max_iterations + start;

  Tensor actions =
    torch::zeros({this->cfg_->env_cfg.num_envs, this->env_->get_action_size()}, this->device_);
  for (unsigned int it = start; it < end; ++it) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Rollout
    {
      torch::NoGradGuard no_grad;
      for (unsigned int i = 0; i < this->cfg_->runner_cfg.num_steps_per_env; ++i) {
        this->train_algorithm_->act(actions, this->observation_buffer_->get_actor_obs(),
                                    this->observation_buffer_->get_critic_obs());
        this->env_->step(this->env_results_, actions);
        this->observation_buffer_->memorize(this->env_results_, actions);

        this->train_algorithm_->process_step(
          this->env_results_.rewards, this->env_results_.terminated, this->env_results_.truncated);

        this->current_reward_sum_ += this->env_results_.rewards;
        this->current_episode_length_ += 1;

        const Tensor& done_ids = (this->env_results_.terminated | this->env_results_.truncated);

        if (done_ids.sum().item<int>() > 0) {
          this->reward_buffer_->push(
            utils::tensor_to_vector<float>(this->current_reward_sum_.index({done_ids}).cpu()));
          this->length_buffer_->push(
            utils::tensor_to_vector<int>(this->current_episode_length_.index({done_ids}).cpu()));

          this->current_reward_sum_.index_put_({done_ids}, 0.0);
          this->current_episode_length_.index_put_({done_ids}, 0);

          this->env_->reset(this->env_results_, done_ids);
          this->observation_buffer_->reset(this->env_results_, done_ids);
        }
      }
      this->collection_time_ =
        std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start_time)
          .count();

      // Learning step
      start_time = std::chrono::high_resolution_clock::now();
      this->train_algorithm_->compute_returns(this->observation_buffer_->get_critic_obs());
    }

    const algorithms::LossMetrics loss_metrics = this->train_algorithm_->update_actor_critic();
    this->learn_time_ =
      std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start_time).count();
    this->current_learning_iteration_ += 1;

    // Logging
    float iteration_time = this->collection_time_ + this->learn_time_;
    int time_steps = this->cfg_->env_cfg.num_envs * this->cfg_->runner_cfg.num_steps_per_env;
    this->total_time_ += iteration_time;
    this->total_time_steps_ += time_steps;

    const ComputationMetrics computation_metrics{time_steps / iteration_time, iteration_time,
                                                 this->collection_time_, this->learn_time_};

    const RewardMetrics reward_metrics{this->reward_buffer_->mean(), this->length_buffer_->mean()};

    std::map<string, float> extra_values;
    const Tensor action_stds = this->train_algorithm_->get_action_std().cpu();
    for (int i = 0; i < action_stds.size(0); ++i) {
      extra_values["action_std_" + std::to_string(i)] = action_stds[i].item<float>();
    }
    extra_values["learning_rate"] = this->train_algorithm_->get_learning_rate();
    const ExtraMetrics extra_metrics{extra_values};

    const TotalMetrics total_metrics{this->total_time_steps_, this->total_time_};

    const TrainMetrics metric{this->current_learning_iteration_,
                              end,
                              computation_metrics,
                              loss_metrics,
                              reward_metrics,
                              total_metrics,
                              extra_metrics};

    this->log_metric_(metric);

    // Save models
    if (it + 1 % this->cfg_->runner_cfg.save_interval == 0)
      this->save_models("/models_" + std::to_string(this->current_learning_iteration_) + ".pt");
  }
  // Save models
  this->save_models("/models_last.pt");
}

void OnPolicyRunner::play() {
  torch::NoGradGuard no_grad;
  this->eval_();
  const auto& policy = this->get_inference_policy();
  this->env_->reset(this->env_results_);
  this->observation_buffer_->reset(this->env_results_);
  while (true) {
    const Tensor actions = policy(this->observation_buffer_->get_actor_obs());
    this->env_->step(this->env_results_, actions);
    this->observation_buffer_->memorize(this->env_results_, actions);
    this->env_->update_render_trajectory(this->env_results_);

    const Tensor& done_ids = (this->env_results_.terminated | this->env_results_.truncated);
    if (done_ids.all().item<bool>()) break;
  }
  this->env_->close();

  std::cout << "-------Rendering-------" << std::endl;
  this->env_->render();
}

void OnPolicyRunner::save_models(const string& name) const {
  torch::serialize::OutputArchive archive;
  this->train_algorithm_->save_models(archive);

  const string run_path = utils::get_run_path(this->cfg_->env_cfg.task);
  archive.save_to(run_path + name);
}

void OnPolicyRunner::load_models(const string& name, const bool& load_optimizer) {
  torch::serialize::InputArchive archive;
  const string run_path = utils::get_run_path(this->cfg_->env_cfg.task);
  archive.load_from(run_path + name);
  this->train_algorithm_->load_models(archive, load_optimizer);
}

void OnPolicyRunner::update_cfg_() {
  unsigned int num_actor_obs = this->observation_buffer_->get_actor_obs_size();
  unsigned int num_critic_obs = this->observation_buffer_->get_critic_obs_size();
  this->cfg_->update(num_actor_obs, num_critic_obs, this->env_->get_action_min(),
                     this->env_->get_action_max());
}

void OnPolicyRunner::initialize_() {
  this->env_->initialize();
  this->env_results_.actor_obs =
    torch::zeros({this->cfg_->env_cfg.num_envs, this->env_->get_actor_obs_size()}, this->device_);
  this->env_results_.critic_obs =
    torch::zeros({this->cfg_->env_cfg.num_envs, this->env_->get_critic_obs_size()}, this->device_);
  this->env_results_.rewards = torch::zeros({this->cfg_->env_cfg.num_envs}, this->device_);
  this->env_results_.terminated =
    torch::zeros({this->cfg_->env_cfg.num_envs},
                 torch::TensorOptions().device(this->device_).dtype(torch::kBool));
  this->env_results_.truncated =
    torch::zeros({this->cfg_->env_cfg.num_envs},
                 torch::TensorOptions().device(this->device_).dtype(torch::kBool));

  this->current_reward_sum_ = torch::zeros({this->cfg_->env_cfg.num_envs}, this->device_);
  this->current_episode_length_ =
    torch::zeros({this->cfg_->env_cfg.num_envs},
                 torch::TensorOptions().device(this->device_).dtype(torch::kInt32));
}

void OnPolicyRunner::log_metric_(const TrainMetrics& metric) const {
  std::cout << metric << std::endl;
  if (this->current_learning_iteration_ < this->cfg_->runner_cfg.logging_warmup) return;
  const std::map<string, float> data = metric.to_dict();
  for (const auto& [key, value] : data)
    this->logger_->add_scalar(key, this->current_learning_iteration_, value);
}

}  // namespace runners