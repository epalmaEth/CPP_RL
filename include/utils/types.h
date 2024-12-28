#pragma once

#include <torch/torch.h>

#include <map>
#include <string>

// Standard types
using string = std::string;

// LibTorch types
using Tensor = torch::Tensor;
using Device = torch::Device;
using ListTensor = std::vector<Tensor>;
using DictTensor = std::map<string, Tensor>;
using DictListTensor = std::map<string, ListTensor>;

// NN types
using NNModule = torch::nn::Module;
using MLP = torch::nn::Sequential;

// Env types
struct StepResult {
  Tensor observation;
  Tensor reward;
  Tensor terminated;
  Tensor truncated;
  DictTensor info;
};