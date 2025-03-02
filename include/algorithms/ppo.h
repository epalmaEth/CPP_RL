#pragma once

#include <torch/torch.h>

#include "configs/configs.h"
#include "modules/actor_critic.h"
#include "storage/rollout.h"
#include "utils/utils.h"

namespace algorithms {

using AdamPointer = std::unique_ptr<torch::optim::Adam>;

struct LossMetrics {
  float actor_loss;
  float critic_loss;
  float entropy_loss;
  float kl_loss;

  LossMetrics(float actor_loss, float critic_loss, float entropy_loss, float kl_loss)
    : actor_loss(actor_loss),
      critic_loss(critic_loss),
      entropy_loss(entropy_loss),
      kl_loss(kl_loss) {}

  friend std::ostream& operator<<(std::ostream& os, const LossMetrics& metrics) {
    os << "\033[1mLoss Metrics:\033[0m" << std::endl;
    os << utils::formatOutput("Actor Loss", metrics.actor_loss) << std::endl;
    os << utils::formatOutput("Critic Loss", metrics.critic_loss) << std::endl;
    os << utils::formatOutput("Entropy Loss", metrics.entropy_loss) << std::endl;
    os << utils::formatOutput("KL Loss", metrics.kl_loss) << std::endl;
    return os;
  }
};

class PPO {
 public:
  PPO(const configs::CfgPointer& cfg, const Device& device);

  void act(Tensor& actions, const Tensor& actor_obs, const Tensor& critic_obs);
  void process_step(const Tensor& rewards, const Tensor& terminated, const Tensor& truncated);
  void compute_returns(const Tensor critic_obs);
  const LossMetrics update_actor_critic();
  const std::function<Tensor(const Tensor&)> get_inference_policy() const {
    return this->actor_critic_->get_inference_policy();
  }
  const Tensor& get_action_std() const { return this->actor_critic_->get_action_std(); };
  float get_learning_rate() const { return this->optimizer_->param_groups()[0].options().get_lr(); }
  void train() { this->actor_critic_->train(); }
  void eval() { this->actor_critic_->eval(); }
  void save_models(torch::serialize::OutputArchive& archive) const;
  void load_models(torch::serialize::InputArchive& archive, const bool& load_optimizer);

 private:
  void initialize_();

  const configs::CfgPointer cfg_;
  modules::ActorCriticPointer actor_critic_;
  storage::RolloutStoragePointer rollout_storage_;
  AdamPointer optimizer_;

  const Device device_;
  storage::Transition transition_;
};

using PPOPointer = std::unique_ptr<algorithms::PPO>;
}  // namespace algorithms