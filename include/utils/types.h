#pragma once

#include <torch/torch.h>
#include <map>
#include <string>

// LibTorch types
using Tensor = torch::Tensor;
using Device = torch::Device;
using TensorDict = std::map<std::string, Tensor>;

// Env types
struct StepResult {
    Tensor observation;
    Tensor reward;
    Tensor terminated;
    Tensor truncated;
    TensorDict info;
};