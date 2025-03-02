#pragma once

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>

#include "types.h"

namespace utils {
inline unsigned int next_power_of_2(const unsigned int& n, const unsigned int& i) {
  if (n == 0) return 0;
  if (n == 1) return 1;
  return 1 << (unsigned int)(std::log2(n - 1) + 1) + i;
}

template <typename T>
inline std::vector<T> tensor_to_vector(const Tensor& tensor) {
  std::vector<T> vec(tensor.numel());
  std::memcpy(vec.data(), tensor.data_ptr<T>(), tensor.numel() * sizeof(T));
  return vec;
}

inline string formatOutput(const string& name, const float& value, const string& unit = "",
                           const int& pad = 35) {
  std::ostringstream oss;
  oss << std::left << std::setw(pad) << name << ":" << std::right << std::setw(pad) << value
      << unit;
  return oss.str();
}

inline unsigned int last_run_id(const string& path) {
  int id = 0;
  const std::regex pattern_regex(R"(run_(\d+))");
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    std::smatch match;
    const string name = entry.path().filename().string();
    if (std::regex_match(name, match, pattern_regex)) id += 1;
  }
  return id;
}

inline string get_run_path(const string& task) {
  const string task_path = "data/" + task;
  int id = last_run_id(task_path);
  return task_path + "/run_" + std::to_string(id);
}

}  // namespace utils