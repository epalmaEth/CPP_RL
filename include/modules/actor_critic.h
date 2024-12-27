#pragma once

#include "distribution.h"
#include "utils/types.h"

namespace modules {

    class Actor : public NNModule {
    public:
        Actor(const int& num_observations, const int& num_actions, const int& depth,
              const double& init_noise_std, const torch::nn::Functional& activation,
              const string& distribution_type, const Device& device);

        Tensor forward(const Tensor& actor_observations, const bool& inference);
        Tensor get_mean() const {return this->distribution_->get_mean();}
        Tensor get_std() const {return this->distribution_->get_std();}
        Tensor get_log_prob(const Tensor& actions) const {return this->distribution_->get_log_prob(actions);}
        Tensor get_entropy() const {return this->distribution_->get_entropy();}
        Tensor get_kl(const DictTensor& old_kl_params) const {return this->distribution_->get_kl(old_kl_params);}
        DictTensor get_kl_params() const {return this->distribution_->get_kl_params();}

    private:
        MLP network_;
        std::shared_ptr<Distribution> distribution_;
    };

    class Critic : public NNModule {
    public:
        Critic(const int& num_observations, const int& depth,
               const torch::nn::Functional& activation, const Device& device);

        Tensor forward(const Tensor& critic_observations) {return this->network_->forward(critic_observations);};

    private:
        MLP network_;
    };

    class ActorCritic : public NNModule {
    public:
        ActorCritic(const int& num_actor_obs, const int& num_critic_obs, 
                    const int& num_actions, const double& init_noise_std,
                    const int& depth_actor, const int& depth_critic, const Device& device,
                    const torch::nn::Functional& activation_actor, const torch::nn::Functional& activation_critic,
                    const std::string& distribution_type = "normal"):
        actor_(num_actor_obs, num_actions, depth_actor, init_noise_std, activation_actor, distribution_type, device),
        critic_(num_critic_obs, depth_critic, activation_critic, device) {}

        Tensor forward(const Tensor& actor_observations, const bool& inference) { return this->actor_.forward(actor_observations, inference);}
        Tensor evaluate(const Tensor& critic_observations) {return this->critic_.forward(critic_observations);}
        Tensor get_action_mean() const {return this->actor_.get_mean();}
        Tensor get_action_std() const {return this->actor_.get_std();}
        Tensor get_actions_log_prob(const Tensor& actions) const {return this->actor_.get_log_prob(actions).sum(-1);}
        Tensor get_entropy() const {return this->actor_.get_entropy().sum(-1);}
        Tensor get_kl(const DictTensor& old_kl_params) const {return this->actor_.get_kl(old_kl_params);}
        DictTensor get_distribution_kl_params() const {return this->actor_.get_kl_params();}

    private:
        Actor actor_;
        Critic critic_;
    };
} // namespace modules