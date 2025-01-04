#pragma once

#include <torch/torch.h>

#include <map>
#include <string>

// Standard types
using string = std::string;

// LibTorch types
using Slice = torch::indexing::Slice;
using Tensor = torch::Tensor;
using Device = torch::Device;
using ListTensor = std::vector<Tensor>;
using DictTensor = std::unordered_map<string, Tensor>;
using DictListTensor = std::unordered_map<string, ListTensor>;

// NN types
using NNModule = torch::nn::Module;
using NN = torch::nn::Sequential;

// Env types
struct StepResult {
  Tensor obs;
  Tensor reward;
  Tensor terminated;
  Tensor truncated;
  DictTensor info;
};