#include "storage/observation_buffer.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

// Structure to hold test parameters.
struct ObsBufferTestParams {
  int num_actor_obs;       // 1 or 2
  int num_critic_obs;      // 1 or 2
  int num_actions;         // 1, or 2
  int num_envs;            // 1 or 2
  int memory_length;       // 1 or 2
  bool store_action;       // false or true
  std::string device_str;  // "cpu" or "gpu"
  bool use_indices;        // false: update all envs, true: update one env (index 0)
};

// Parameterized test fixture.
class ObservationBufferParameterizedTest : public ::testing::TestWithParam<ObsBufferTestParams> {
 protected:
  void SetUp() override {
    // Get the test parameters.
    const auto& params = GetParam();

    // Create a dummy config.
    const configs::EnvCfg env_cfg{.num_envs = params.num_envs};
    const configs::RunnerCfg runner_cfg{.observation_memory_length = params.memory_length,
                                        .observation_memory_store_action = params.store_action};
    const configs::PPOCfg ppo_cfg{};
    const configs::ActorCfg actor_cfg{};
    const configs::CriticCfg critic_cfg{};
    this->cfg_ =
      std::make_shared<configs::Cfg>(env_cfg, runner_cfg, ppo_cfg, actor_cfg, critic_cfg);

    this->num_actor_obs_ = params.num_actor_obs;
    this->num_critic_obs_ = params.num_critic_obs;
    this->num_actions_ = params.num_actions;
    this->use_indices_ = params.use_indices;

    // Set device.
    if (params.device_str == "gpu") {
      if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available, skipping GPU test.";
      }
      device_ = Device(torch::kCUDA);
    } else {
      device_ = Device(torch::kCPU);
    }
  }

  configs::CfgPointer cfg_;
  int num_actor_obs_;
  int num_critic_obs_;
  int num_actions_;
  bool use_indices_;
  torch::Device device_;
};

// Test that reset fills the buffer correctly.
// If "use_indices_" is true, only environment 0 should be updated; others should remain zero.
TEST_P(ObservationBufferParameterizedTest, ResetFillsBufferWithExtendedObs) {
  torch::manual_seed(0);
  storage::ObservationBuffer obs_buffer(cfg_, num_actor_obs_, num_critic_obs_, num_actions_,
                                        device_);

  // Create dummy actor and critic observations.
  const auto actor_obs =
    torch::rand({num_envs_, num_actor_obs_}, torch::TensorOptions().device(device_));
  const auto critic_obs =
    torch::rand({num_envs_, num_critic_obs_}, torch::TensorOptions().device(device_));

  const env::Results results{
    .actor_obs = actor_obs,
    .critic_obs = critic_obs,
  };

  // Prepare indices: if use_indices_ is true, update only env 0 via a boolean mask.
  torch::Tensor indices;
  if (use_indices_) {
    indices = torch::zeros({num_envs_}, torch::TensorOptions().dtype(torch::kBool).device(device_));
    indices[0] = true;
    obs_buffer.reset(results, indices);
  } else {
    obs_buffer.reset(results);
  }

  // Build the expected extended observation for updated envs.
  // When store_action is true, we expect the original obs concatenated with
  // zeros.
  torch::Tensor expected_actor_obs, expected_critic_obs;
  if (cfg_->runner_cfg.observation_memory_store_action) {
    // Number of environments updated.
    int n_update = use_indices_ ? 1 : num_envs_;
    auto zeros_actions = torch::zeros({n_update, num_actions_},
                                      torch::TensorOptions().dtype(torch::kFloat).device(device_));
    if (use_indices_) {
      expected_actor_obs = torch::cat({actor_obs.index({0}).unsqueeze(0), zeros_actions}, 1);
      expected_critic_obs = torch::cat({critic_obs.index({0}).unsqueeze(0), zeros_actions}, 1);
    } else {
      expected_actor_obs = torch::cat({actor_obs, zeros_actions}, 1);
      expected_critic_obs = torch::cat({critic_obs, zeros_actions}, 1);
    }
  } else {
    // Otherwise, the expected extended observation is just the original observation.
    expected_actor_obs = use_indices_ ? actor_obs.index({0}).unsqueeze(0) : actor_obs;
    expected_critic_obs = use_indices_ ? critic_obs.index({0}).unsqueeze(0) : critic_obs;
  }

  // Get the internal buffer view with shape [num_envs, memory_length, extended_dim].
  const auto buffer_actor_obs = obs_buffer.get_actor_obs().view({num_envs_, memory_length_, -1});
  const auto buffer_critic_obs = obs_buffer.get_critic_obs().view({num_envs_, memory_length_, -1});

  EXPECT_EQ(buffer_actor_obs.sizes, expected_actor_obs.sizes()) << "Mismatch in actor obs size";
  EXPECT_EQ(buffer_critic_obs.sizes, expected_critic_obs.sizes()) << "Mismatch in critic obs size";

  EXPECT_TRUE(torch::allclose(buffer_actor_obs, expected_actor_obs)) << "Mismatch in actor obs";
  EXPECT_TRUE(torch::allclose(buffer_critic_obs, expected_critic_obs)) << "Mismatch in critic obs";
}

// Test that memorize shifts (rolls) the buffer and updates the last time step.
TEST_P(ObservationBufferParameterizedTest, MemorizeRollsBufferAndUpdatesLastStep) {
  torch::manual_seed(0);
  storage::ObservationBuffer obs_buffer(cfg_, num_actor_obs_, num_critic_obs_, num_actions_,
                                        device_);

  // Reset the buffer with a baseline observation.
  const auto reset_actor_obs =
    torch::rand({num_envs_, num_actor_obs_}, torch::TensorOptions().device(device_));
  const auto reset_critic_obs =
    torch::rand({num_envs_, num_critic_obs_}, torch::TensorOptions().device(device_));
  const env::Results reset_results{
    .actor_obs = reset_actor_obs,
    .critic_obs = reset_critic_obs,
  };

  obs_buffer.reset(reset_results);

  // Prepare new observations and actions.
  const auto actor_obs =
    torch::rand({num_envs_, num_actor_obs_}, torch::TensorOptions().device(device_));
  const auto critic_obs =
    torch::rand({num_envs_, num_critic_obs_}, torch::TensorOptions().device(device_));
  const auto actions =
    torch::rand({num_envs_, num_actions_}, torch::TensorOptions().device(device_));

  env::Results new_results{
    .actor_obs = actor_obs,
    .critic_obs = critic_obs,
  };

  // Call memorize.
  obs_buffer.memorize(new_results, actions);

  // Determine expected new extended observations.
  torch::Tensor expected_new_actor_obs, expected_new_critic_obs;
  if (cfg_->runner_cfg.observation_memory_store_action) {
    expected_new_actor_obs = torch::cat({actor_obs, actions}, 1);
    expected_new_critic_obs = torch::cat({critic_obs, actions}, 1);
  } else {
    expected_new_actor_obs = actor_obs;
    expected_new_critic_obs = critic_obs;
  }

  // The last time slot in each environment should now match the expected new extended observation.
  const auto buffer_actor_obs = obs_buffer.get_actor_obs().view({num_envs_, memory_length_, -1});
  const auto buffer_critic_obs = obs_buffer.get_critic_obs().view({num_envs_, memory_length_, -1});

  EXPECT_EQ(buffer_actor_obs.sizes, expected_new_actor_obs.sizes()) << "Mismatch in actor obs size";
  EXPECT_EQ(buffer_critic_obs.sizes, expected_new_critic_obs.sizes())
    << "Mismatch in critic obs size";
  EXPECT_TRUE(torch::allclose(buffer_actor_obs, expected_new_actor_obs))
    << "Memorize did not update actor obs correctly";
  EXPECT_TRUE(torch::allclose(buffer_critic_obs, expected_new_critic_obs))
    << "Memorize did not update critic obs correctly";

  if (memory_length_ > 1) {
    // Verify the shifting`of the buffer.
    EXPECT_TRUE(torch::allclose(buffer_actor_obs.slice(1, 0, memory_length_ - 1),
                                buffer_actor_obs.slice(1, 1, memory_length_)))
      << "Shifted actor observations do not match.";
    EXPECT_TRUE(torch::allclose(buffer_critic_obs.slice(1, 0, memory_length_ - 1),
                                buffer_critic_obs.slice(1, 1, memory_length_)))
      << "Shifted critic observations do not match.";
  }
}

// Instantiate tests using Cartesian product of all parameter sets.
INSTANTIATE_TEST_SUITE_P(
  ObservationBufferTests, ObservationBufferParameterizedTest,
  ::testing::Combine(
    // num_actor_obs: 1 or 2
    ::testing::Values(1, 2),
    // num_critic_obs: 1 or 2
    ::testing::Values(1, 2),
    // num_actions: 1 or 2
    ::testing::Values(1, 2),
    // num_envs: 1 or 2
    ::testing::Values(1, 2),
    // memory_length: 1 or 2
    ::testing::Values(1, 2),
    // store_action: false or true
    ::testing::Values(false, true),
    // device: "cpu" or "gpu"
    ::testing::Values(std::string("cpu"), std::string("gpu")),
    // indices: false (update all) or true (update only one env)
    ::testing::Values(false, true)),
  [](const ::testing::TestParamInfo<std::tuple<int, int, int, int, int, bool, std::string, bool>>&
       info) {
    int num_actor_obs, num_critic_obs, num_actions, num_envs, memory_length;
    bool store_action, use_indices;
    std::string device_str;
    std::tie(num_actor_obs, num_critic_obs, num_actions, num_envs, memory_length, store_action,
             device_str, use_indices) = info.param;
    std::stringstream ss;
    ss << "A" << num_actor_obs << "_C" << num_critic_obs << "_Act" << num_actions << "_E"
       << num_envs << "_M" << memory_length << "_" << (store_action ? "Store" : "NoStore") << "_"
       << device_str << "_" << (use_indices ? "Indices" : "NoIndices");
    return ss.str();
  });

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
