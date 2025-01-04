#pragma once

#include "configs/configs.h"
#include "torch/torch.h"
#include "utils/types.h"

namespace modules {

using NormalizerCfg = configs::NormalizerCfg;

class Normalizer : public torch::nn::Module {
 public:
  Normalizer() = default;

  ~Normalizer() = default;

  virtual Tensor forward(const Tensor& observations) = 0;
  virtual Tensor normalize(const Tensor& observations) const = 0;
  virtual Tensor denormalize(const Tensor& observations) const = 0;

  void train() { this->inference_mode_ = false; }
  void eval() { this->inference_mode_ = true; }

 protected:
  bool inference_mode_ = false;
};

class IdentityNormalizer : public Normalizer {
 public:
  // Inherit the constructor from the base class
  using Normalizer::Normalizer;

  Tensor forward(const Tensor& observations) override { return observations; }
  Tensor normalize(const Tensor& observations) const override {
    return observations;
  }
  Tensor denormalize(const Tensor& observations) const override {
    return observations;
  }
};

class EmpiricalNormalizer : public Normalizer {
 public:
  EmpiricalNormalizer(const NormalizerCfg& cfg) {
    this->mean_ = torch::zeros({cfg.num_inputs});
    this->var_ = torch::ones({cfg.num_inputs});

    this->register_buffer("mean", this->mean_);
    this->register_buffer("var", this->var_);
  }

  Tensor forward(const Tensor& observations) override;
  void update(const Tensor& observations);
  Tensor normalize(const Tensor& observations) const override;
  Tensor denormalize(const Tensor& observations) const override;

 private:
  unsigned int count_ = 0;

  Tensor mean_;
  Tensor var_;
};

class NormalizerFactory {
 public:
  static std::shared_ptr<Normalizer> create(const NormalizerCfg& cfg) {
    if (cfg.type == "identity") return std::make_shared<IdentityNormalizer>();
    if (cfg.type == "empirical")
      return std::make_shared<EmpiricalNormalizer>(cfg);
    throw std::invalid_argument("Unknown normalizer type: " + cfg.type);
  }
};

}  // namespace modules