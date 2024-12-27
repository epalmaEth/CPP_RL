#include "modules/distribution.h"

#include <torch/torch.h>

namespace modules {

    using Slice = torch::indexing::Slice;

    void Normal::update(const Tensor& hidden_output) {
        this->mean_ = hidden_output;
    }

    Tensor Normal::get_sample() const {
        return at::normal(this->mean_, this->std_.expand_as(this->mean_));
    }

    Tensor Normal::get_log_prob(const Tensor& actions) const {
        const Tensor& var = this->std_.square();
        return -0.5*(torch::log(2.*M_PI*var) + (actions - this->mean_).square()/var);
    }

    Tensor Normal::get_entropy() const {
        const Tensor& var = this->std_.square();
        return 0.5*(1 + torch::log(2.*M_PI*var));
    }

    Tensor Normal::get_kl(const DictTensor& old_kl_params) const {
        const Tensor& old_mean = old_kl_params.at("mean");
        const Tensor& old_std = old_kl_params.at("std");
        const Tensor& new_std = this->std_;
        const Tensor& new_mean = this->mean_;

        return torch::sum(
            torch::log(new_std/old_std) + 
            (old_std.square() + (old_mean - new_mean).square())/
            (2.*new_std.square()) - 
            0.5, 
            /*dim=*/-1
        );
    }

    DictTensor Normal::get_kl_params() const {
        return {
            {"mean", this->mean_.clone()},
            {"std", this->std_.clone()}
        };
    }

    void Beta::update(const Tensor& hidden_output) {
        this->mean_ = hidden_output;
    
        const Tensor& total = torch::relu(this->mean_*(1.-this->mean_) / this->std_.expand_as(this->mean_).square() - 1.) + 1e-07;
        this->alpha_ = this->mean_*total;
        this->beta_ = (1.-this->mean_)*total;
    }

    Tensor Beta::get_sample() const {
        return at::_sample_dirichlet(torch::stack({this->alpha_, this->beta_}, /*dim=*/-1));
    }

    Tensor Beta::get_log_prob(const Tensor& actions) const {
        return (this->alpha_ - 1.)*torch::log(actions) + 
        (this->beta_ - 1.)*torch::log(1. - actions) - 
        torch::lgamma(this->alpha_ + this->beta_) + 
        torch::lgamma(this->alpha_) + 
        torch::lgamma(this->beta_);
    }

    Tensor Beta::get_entropy() const {
        const Tensor& alpha_beta = this->alpha_ + this->beta_;
        return torch::lgamma(alpha_beta) - torch::lgamma(this->alpha_) - torch::lgamma(this->beta_) + 
        (this->alpha_ - 1.)*(torch::digamma(this->alpha_) - torch::digamma(alpha_beta)) + 
        (this->beta_ - 1.)*(torch::digamma(this->beta_) - torch::digamma(alpha_beta));
    }

    Tensor Beta::get_kl(const DictTensor& old_kl_params) const {
        const Tensor& alpha = this->alpha_;
        const Tensor& beta = this->beta_;
        const Tensor& old_alpha = old_kl_params.at("alpha");
        const Tensor& old_beta = old_kl_params.at("beta");

        return torch::sum(
            torch::lgamma(old_alpha + old_beta) - torch::lgamma(old_alpha) - torch::lgamma(old_beta) + 
            (old_alpha - alpha)*(torch::digamma(old_alpha) - torch::digamma(old_alpha + old_beta)) + 
            (old_beta - beta)*(torch::digamma(old_beta) - torch::digamma(old_alpha + old_beta)), 
            /*dim=*/-1
        );
    }

    DictTensor Beta::get_kl_params() const {
        return {
            {"alpha", this->alpha_.clone()},
            {"beta", this->beta_.clone()}
        };
    }

} // namespace modules