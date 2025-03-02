#pragma once

#include "beta.h"
#include "modules/distribution.h"
#include "normal.h"
#include "utils/types.h"

namespace modules {

class DistributionFactory {
 public:
  static const std::shared_ptr<Distribution> create(const configs::DistributionCfg& cfg) {
    if (cfg.type == "normal") return std::make_shared<Normal>(cfg);
    if (cfg.type == "beta") return std::make_shared<Beta>(cfg);
    throw std::invalid_argument("Unknown distribution type: " + cfg.type);
  }
};

}  // namespace modules