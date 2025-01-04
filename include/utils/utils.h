#pragma once

#include <cmath>

#include "utils/types.h"

namespace utils {
inline unsigned int next_power_of_2(const unsigned int& n,
                                    const unsigned int& i) {
  if (n <= 0) return 0;
  if (n == 1) return 1;
  return 1 << (unsigned int)(std::log2(n - 1) + 1) + i;
}
}  // namespace utils