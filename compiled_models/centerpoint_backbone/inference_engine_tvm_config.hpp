// Copyright 2021 Arm Limited and Contributors.
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

#include "pipeline.hpp"

#ifndef CENTERPOINT_BACKBONE_INFERENCE_ENGINE_TVM_CONFIG_HPP_  // NOLINT
#define CENTERPOINT_BACKBONE_INFERENCE_ENGINE_TVM_CONFIG_HPP_

namespace model_zoo
{
namespace perception
{
namespace lidar_obstacle_detection
{
namespace centerpoint_backbone
{
namespace onnx_centerpoint_backbone
{

static const tvm_utility::pipeline::InferenceEngineTVMConfig config {
  {
    0,
    0,
    0
  },  // modelzoo_version

  "centerpoint_backbone",  // network_name
  "cuda",  // network_backend

  "./deploy_lib.so",  //network_module_path
  "./deploy_graph.json",  // network_graph_path
  "./deploy_param.params",  // network_params_path

  kDLCUDA,  // tvm_device_type
  0,  // tvm_device_id

  {
    {"spatial_features", kDLFloat, 32, 1, {1, 32, 560, 560}}
  },  // network_inputs

  {
    {"heatmap", kDLFloat, 32, 1, {1, 5, 560, 560}},
    {"reg", kDLFloat, 32, 1, {1, 2, 560, 560}},
    {"height", kDLFloat, 32, 1, {1, 1, 560, 560}},
    {"dim", kDLFloat, 32, 1, {1, 3, 560, 560}},
    {"rot", kDLFloat, 32, 1, {1, 2, 560, 560}},
    {"vel", kDLFloat, 32, 1, {1, 2, 560, 560}}
  }  // network_outputs
};

}  // namespace onnx_centerpoint_backbone
}  // namespace centerpoint_backbone
}  // namespace lidar_obstacle_detection
}  // namespace perception
}  // namespace model_zoo
#endif  // CENTERPOINT_BACKBONE_INFERENCE_ENGINE_TVM_CONFIG_HPP_  // NOLINT
