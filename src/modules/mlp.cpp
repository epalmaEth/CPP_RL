#include "modules/mlp.h"

namespace modules {
MLP::MLP(const configs::MLPCfg& cfg) {
  this->network_ = NN();
  this->network_->push_back(torch::nn::Linear(cfg.num_inputs, cfg.width));
  this->add_activation_(cfg.activation);
  for (unsigned int i = 0; i < cfg.depth; i++) {
    this->network_->push_back(torch::nn::Linear(cfg.width, cfg.width));
    this->add_activation_(cfg.activation);
  }
  this->network_->push_back(torch::nn::Linear(cfg.width, cfg.num_outputs));
  this->register_module("network", this->network_);
}

void MLP::add_activation_(const string& activation) {
  if (activation == "relu")
    this->network_->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
  else if (activation == "elu")
    this->network_->push_back(torch::nn::ELU(torch::nn::ELUOptions().inplace(true)));
  else if (activation == "tanh")
    this->network_->push_back(torch::nn::Tanh());
  else if (activation == "sigmoid")
    this->network_->push_back(torch::nn::Sigmoid());
  else
    throw std::invalid_argument("Invalid activation function");
}
}  // namespace modules