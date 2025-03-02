#include <torch/cuda.h>
#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <iostream>

#include "configs/load_yaml.h"
#include "env/env.h"
#include "runners/on_policy_runner.h"
#include "utils/types.h"
#include "utils/utils.h"

void check_task_folder(const string& task) {
  const string task_path = "data/" + task;
  if (!std::filesystem::exists(task_path)) {
    std::cout << "Error: Task folder '" << task_path << "' does not exist. Train once first."
              << std::endl;
    std::exit(1);
  }
}

void check_run_folder(const string& task, const int& run_id) {
  const string run_path = utils::get_run_path(task);
  if (!std::filesystem::exists(run_path)) {
    std::cout << "Error: Run folder '" << run_path
              << "' does not exist. Set available training run to play" << std::endl;
    std::exit(1);
  }
}

void create_run_folder(const string& task) {
  const string task_folder = "data/" + task;
  std::filesystem::create_directories(task_folder);
  int id = utils::last_run_id(task_folder) + 1;
  const string run_folder = task_folder + "/run_" + std::to_string(id);
  std::filesystem::create_directories(run_folder);
}

void copy_yaml(const string& task) {
  const string run_path = utils::get_run_path(task);
  std::filesystem::copy("yaml/train.yaml", run_path + "/config.yaml");
}

int main(int argc, char* argv[]) {
  bool playing = string(argv[1]) == "play";
  const string& task = string(argv[2]);

  const Device& device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  std::cout << "Device: " << device << std::endl;

  if (playing)
    check_task_folder(task);
  else
    create_run_folder(task);

  std::cout << "-------Loading Cfg-------" << std::endl;
  const configs::CfgPointer& cfg = configs::load_config(task, playing);

  std::cout << "-------Creating Runner-------" << std::endl;
  const runners::RunnerPointer& runner =
    std::make_unique<runners::OnPolicyRunner>(task, cfg, device);

  if (playing) {
    check_run_folder(task, cfg->env_cfg.run_id);
    std::cout << "-------Loading Model-------" << std::endl;
    runner->load_models("/models_last.pt");
    std::cout << "-------Play-------" << std::endl;
    runner->play();
  } else {
    std::cout << "-------Train-------" << std::endl;
    runner->learn();
    std::cout << "-------Copy Yaml-------" << std::endl;
    copy_yaml(task);
  }
}
