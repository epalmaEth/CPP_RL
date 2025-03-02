#pragma once

#include "empirical.h"
#include "identity.h"
#include "modules/normalizer.h"
#include "utils/types.h"

namespace modules {

class NormalizerFactory {
 public:
  static std::shared_ptr<Normalizer> create(const configs::NormalizerCfg& cfg) {
    if (cfg.type == "identity") return std::make_shared<IdentityNormalizer>();
    if (cfg.type == "empirical") return std::make_shared<EmpiricalNormalizer>(cfg);
    throw std::invalid_argument("Unknown normalizer type: " + cfg.type);
  }
};

}  // namespace modules