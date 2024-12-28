#pragma once

#include "torch/torch.h"
#include "utils/types.h"

namespace modules {

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
  EmpiricalNormalizer(const int& num_observations, const Device& device)
      : device_(device) {
    this->mean_ = torch::zeros({num_observations}).to(device_);
    this->var_ = torch::ones({num_observations}).to(device_);

    this->register_buffer("mean", this->mean_);
    this->register_buffer("var", this->var_);
  }

  Tensor forward(const Tensor& observations) override;
  void update(const Tensor& observations);
  Tensor normalize(const Tensor& observations) const override;
  Tensor denormalize(const Tensor& observations) const override;

 private:
  int count_ = 0;
  Device device_;

  Tensor mean_;
  Tensor var_;
};

class NormalizerFactory {
 public:
  static std::unique_ptr<Normalizer> create(const std::string& normalizer_type,
                                            const int& num_observations,
                                            const torch::Device& device) {
    if (normalizer_type == "identity") {
      return std::make_unique<IdentityNormalizer>();
    } else if (normalizer_type == "empirical") {
      return std::make_unique<EmpiricalNormalizer>(num_observations, device);
    } else {
      throw std::invalid_argument("Unknown normalizer type: " +
                                  normalizer_type);
    }
  }
};

}  // namespace modules