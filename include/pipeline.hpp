// Copyright 2021-2022 Arm Limited and Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TVM_UTILITY__PIPELINE_HPP_
#define TVM_UTILITY__PIPELINE_HPP_

#include <tvm_vendor/dlpack/dlpack.h>
#include <tvm_vendor/tvm/runtime/c_runtime_api.h>
#include <tvm_vendor/tvm/runtime/module.h>
#include <tvm_vendor/tvm/runtime/packed_func.h>
#include <tvm_vendor/tvm/runtime/registry.h>

#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tvm_utility
{

/**
 * @brief Possible version status for a neural network.
 */
enum class Version {
  OK,
  Unknown,
  Untested,
  Unsupported,
};

namespace pipeline
{

class TVMArrayContainer
{
public:
  TVMArrayContainer() = default;

  TVMArrayContainer(
    std::vector<int64_t> shape, DLDataTypeCode dtype_code, int32_t dtype_bits, int32_t dtype_lanes,
    DLDeviceType device_type, int32_t device_id)
  {
    TVMArrayHandle x{};
    TVMArrayAlloc(
      &shape[0], static_cast<int>(shape.size()), dtype_code, dtype_bits, dtype_lanes, device_type,
      device_id, &x);
    handle_ = std::make_shared<TVMArrayHandle>(x);
  }

  TVMArrayHandle getArray() const { return *handle_.get(); }

private:
  std::shared_ptr<TVMArrayHandle> handle_{nullptr, [](TVMArrayHandle ptr) {
                                            if (ptr) {
                                              TVMArrayFree(ptr);
                                            }
                                          }};
};

using TVMArrayContainerVector = std::vector<TVMArrayContainer>;

/**
 * @class PipelineStage
 * @brief Base class for all types of pipeline stages.
 *
 * @tparam InputType The datatype of the input of the pipeline stage.
 * @tparam OutputType The datatype of the output from the pipeline stage.
 */
template <class InputType, class OutputType>
class PipelineStage
{
public:
  /**
   * @brief Execute the pipeline stage
   *
   * @param input The data to push into the pipeline stage. The pipeline stage
   * should not modify the input data.
   * @return The output of the pipeline
   */
  virtual OutputType schedule(const InputType & input) = 0;
  InputType input_type_indicator_;
  OutputType output_type_indicator_;
};

/**
 * @class InferenceEngine
 * @brief Pipeline stage in charge of machine learning inference.
 */
class InferenceEngine : public PipelineStage<TVMArrayContainerVector, TVMArrayContainerVector>
{
};

// NetworkNode
typedef struct
{
  // Node name
  std::string node_name;

  // Network data type configurations
  DLDataTypeCode tvm_dtype_code;
  int32_t tvm_dtype_bits;
  int32_t tvm_dtype_lanes;

  // Shape info
  std::vector<int64_t> node_shape;
} NetworkNode;

typedef struct
{
  // Network info
  std::array<char, 3> modelzoo_version;
  std::string network_name;
  std::string network_backend;

  // Network files
  std::string network_module_path;
  std::string network_graph_path;
  std::string network_params_path;

  // Inference hardware configuration
  DLDeviceType tvm_device_type;
  int32_t tvm_device_id;

  // Network inputs
  std::vector<NetworkNode> network_inputs;

  // Network outputs
  std::vector<NetworkNode> network_outputs;
} InferenceEngineTVMConfig;

class InferenceEngineTVM : public InferenceEngine
{
public:
  explicit InferenceEngineTVM(const InferenceEngineTVMConfig & config, const std::string & pkg_name)
  : config_(config)
  {
    // Get full network path
    std::string network_prefix = pkg_name + "/models/" + config.network_name + "/";
    std::string network_module_path = network_prefix + config.network_module_path;
    std::string network_graph_path = network_prefix + config.network_graph_path;
    std::string network_params_path = network_prefix + config.network_params_path;

    // Load compiled functions
    std::ifstream module(network_module_path);
    if (!module.good()) {
      throw std::runtime_error(
        "File " + network_module_path + " specified in inference_engine_tvm_config.hpp not found");
    }
    module.close();
    tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile(network_module_path);

    // Load json graph
    std::ifstream json_in(network_graph_path, std::ios::in);
    if (!json_in.good()) {
      throw std::runtime_error(
        "File " + network_graph_path + " specified in inference_engine_tvm_config.hpp not found");
    }
    std::string json_data(
      (std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();

    // Load parameters from binary file
    std::ifstream params_in(network_params_path, std::ios::binary);
    if (!params_in.good()) {
      throw std::runtime_error(
        "File " + network_params_path + " specified in inference_engine_tvm_config.hpp not found");
    }
    std::string params_data(
      (std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();

    // Parameters need to be in TVMByteArray format
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();

    // Create tvm runtime module
    tvm::runtime::Module runtime_mod = (*tvm::runtime::Registry::Get("tvm.graph_executor.create"))(
      json_data, mod, static_cast<uint32_t>(config.tvm_device_type), config.tvm_device_id);

    // Load parameters
    auto load_params = runtime_mod.GetFunction("load_params");
    load_params(params_arr);

    // Get set_input function
    set_input = runtime_mod.GetFunction("set_input");

    // Get the function which executes the network
    execute = runtime_mod.GetFunction("run");

    // Get the function to get output data
    get_output = runtime_mod.GetFunction("get_output");

    for (auto & output_config : config.network_outputs) {
      output_.push_back(TVMArrayContainer(
        output_config.node_shape, output_config.tvm_dtype_code, output_config.tvm_dtype_bits,
        output_config.tvm_dtype_lanes, config.tvm_device_type, config.tvm_device_id));
    }
  }

  TVMArrayContainerVector schedule(const TVMArrayContainerVector & input)
  {
    // Set input(s)
    for (uint32_t index = 0; index < input.size(); ++index) {
      if (input[index].getArray() == nullptr) {
        throw std::runtime_error("input variable is null");
      }
      set_input(config_.network_inputs[index].node_name.c_str(), input[index].getArray());
    }

    // Execute the inference
    execute();

    // Get output(s)
    for (uint32_t index = 0; index < output_.size(); ++index) {
      if (output_[index].getArray() == nullptr) {
        throw std::runtime_error("output variable is null");
      }
      get_output(index, output_[index].getArray());
    }
    return output_;
  }

  /**
   * @brief Get version information from the config structure and check if there is a mismatch
   *        between the supported version(s) and the actual version.
   * @param[in] version_from Earliest supported model version.
   * @return The version status.
   */
  Version version_check(const std::array<char, 3> & version_from) const
  {
    auto x{config_.modelzoo_version[0]};
    auto y{config_.modelzoo_version[1]};
    Version ret{Version::OK};

    if (x == 0) {
      ret = Version::Unknown;
    } else if (x > version_up_to[0] || (x == version_up_to[0] && y > version_up_to[1])) {
      ret = Version::Untested;
    } else if (x < version_from[0] || (x == version_from[0] && y < version_from[1])) {
      ret = Version::Unsupported;
    }

    return ret;
  }

private:
  InferenceEngineTVMConfig config_;
  TVMArrayContainerVector output_;
  tvm::runtime::PackedFunc set_input;
  tvm::runtime::PackedFunc execute;
  tvm::runtime::PackedFunc get_output;
  // Latest supported model version.
  const std::array<char, 3> version_up_to{2, 1, 0};
};


}  // namespace pipeline
}  // namespace tvm_utility
#endif  // TVM_UTILITY__PIPELINE_HPP_
