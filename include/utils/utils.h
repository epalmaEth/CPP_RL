#pragma once

#include <cmath>

#include "utils/types.h"

namespace utils {
int next_power_of_2(const int& n) {
  if (n <= 0) return 0;
  if (n == 1) return 1;
  return 1 << (int)(std::log2(n - 1) + 1);
}

MLP create_MLP(const int& input_size, const int& output_size, const int& width,
               const int& depth, const torch::nn::Functional& activation) {
  MLP network;
  network->push_back(torch::nn::Linear(input_size, width));
  network->push_back(activation);
  for (int i = 0; i < depth - 1; i++) {
    network->push_back(torch::nn::Linear(width, width));
    network->push_back(activation);
  }
  network->push_back(torch::nn::Linear(width, output_size));
  return network;
}
}  // namespace utils