#pragma once

#include <torch/torch.h>
#include <map>
#include <string>

// LibTorch types
using Tensor = torch::Tensor;
using Device = torch::Device;
using ListTensor = std::vector<torch::Tensor>;
using DictTensor = std::map<std::string, Tensor>;
using DictListTensor = std::map<std::string, std::vector<torch::Tensor>>;

// Env types
struct StepResult {
    Tensor observation;
    Tensor reward;
    Tensor terminated;
    Tensor truncated;
    DictTensor info;
};