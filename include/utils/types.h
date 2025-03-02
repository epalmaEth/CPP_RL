#pragma once

#include <torch/torch.h>

#include <map>
#include <string>

// Standard types
using string = std::string;

// LibTorch types
using Slice = torch::indexing::Slice;
using Device = torch::Device;
using Tensor = torch::Tensor;
using TensorTuple = std::pair<Tensor, Tensor>;
using ListTensor = std::vector<Tensor>;
using DictTensor = std::unordered_map<string, Tensor>;
using DictListTensor = std::unordered_map<string, ListTensor>;

// NN types
using NNModule = torch::nn::Module;
using NN = torch::nn::Sequential;

// Infinity
constexpr float POS_INF_F = std::numeric_limits<float>::infinity();
constexpr float NEG_INF_F = -std::numeric_limits<float>::infinity();
constexpr int POS_INF_I = std::numeric_limits<int>::max();
constexpr int NEG_INF_I = std::numeric_limits<int>::min();

// Small
constexpr float EPS = std::numeric_limits<float>::epsilon();
