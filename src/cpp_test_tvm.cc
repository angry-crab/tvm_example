#include <cpp_test_tvm.hpp>
#include <pipeline.hpp>
#include <centerpoint_encoder/inference_engine_tvm_config.hpp>
#include <centerpoint_backbone/inference_engine_tvm_config.hpp>
#include <centerpoint_backbone/preprocessing_inference_engine_tvm_config.hpp>
#include <centerpoint_config.hpp>

#include <cstdio>
#include <fstream>

auto config_en = model_zoo::perception::lidar_obstacle_detection::centerpoint_encoder::
  onnx_centerpoint_encoder::config;
auto config_bk = model_zoo::perception::lidar_obstacle_detection::centerpoint_backbone::
  onnx_centerpoint_backbone::config;
auto config_scatter = model_zoo::perception::lidar_obstacle_detection::centerpoint_backbone::
  onnx_centerpoint_backbone::preprocessing::config;

using VE_PrePT = VoxelEncoderPreProcessor;
using IET = tvm_utility::pipeline::InferenceEngineTVM;
using BNH_PostPT = BackboneNeckHeadPostProcessor;
using TSE = TVMScatterIE;
using TSP = tvm_utility::pipeline::TowStagePipeline<VE_PrePT, IET, TSE, BNH_PostPT>;

TVMScatterIE::TVMScatterIE(
  tvm_utility::pipeline::InferenceEngineTVMConfig config, const std::string & pkg_name,
  const std::string & function_name)
: config_(config)
{
  std::string base = "/home/xinyuwang/adehome/tvm_latest/tvm_example";
  std::string network_prefix =
    base + "/compiled_models/" + config.network_name + "/";
  std::string network_module_path = network_prefix + config.network_module_path;

  std::ifstream module(network_module_path);
  if (!module.good()) {
    throw std::runtime_error("File " + network_module_path + " specified in config.hpp not found");
  }
  module.close();
  tvm::runtime::Module runtime_mod = tvm::runtime::Module::LoadFromFile(network_module_path);

  scatter_function = runtime_mod.GetFunction(function_name);

  for (auto & output_config : config.network_outputs) {
    output_.push_back(TVMArrayContainer(
      output_config.node_shape, output_config.tvm_dtype_code, output_config.tvm_dtype_bits,
      output_config.tvm_dtype_lanes, config.tvm_device_type, config.tvm_device_id));
  }
}

TVMArrayContainerVector TVMScatterIE::schedule(const TVMArrayContainerVector & input)
{
  scatter_function(input[0].getArray(), coords_.getArray(), output_[0].getArray());

  return output_;
}

VoxelEncoderPreProcessor::VoxelEncoderPreProcessor(
  const tvm_utility::pipeline::InferenceEngineTVMConfig & config,
  const centerpoint::CenterPointConfig & config_mod)
: max_voxel_size(config.network_inputs[0].node_shape[0]),
  max_point_in_voxel_size(config.network_inputs[0].node_shape[1]),
  encoder_in_feature_size(config.network_inputs[0].node_shape[2]),
  input_datatype_bytes(config.network_inputs[0].tvm_dtype_bits / 8),
  config_detail(config_mod)
{
  encoder_in_features.resize(max_voxel_size * max_point_in_voxel_size * encoder_in_feature_size);
  // Allocate input variable
  std::vector<int64_t> shape_x{1, max_voxel_size, max_point_in_voxel_size, encoder_in_feature_size};
  TVMArrayContainer x{
    shape_x,
    config.network_inputs[0].tvm_dtype_code,
    config.network_inputs[0].tvm_dtype_bits,
    config.network_inputs[0].tvm_dtype_lanes,
    config.tvm_device_type,
    config.tvm_device_id};
  output = x;
}

TVMArrayContainerVector VoxelEncoderPreProcessor::schedule(const std::vector<float> & voxel_inputs)
{
  // generate encoder_in_features from the voxels
  // generateFeatures(
  //   voxel_inputs.features, voxel_inputs.num_points_per_voxel, voxel_inputs.coords,
  //   voxel_inputs.num_voxels, config_detail, encoder_in_features);

  TVMArrayCopyFromBytes(
    output.getArray(), encoder_in_features.data(),
    max_voxel_size * max_point_in_voxel_size * encoder_in_feature_size * input_datatype_bytes);

  return {output};
}

BackboneNeckHeadPostProcessor::BackboneNeckHeadPostProcessor(
  const tvm_utility::pipeline::InferenceEngineTVMConfig & config,
  const centerpoint::CenterPointConfig & config_mod)
: output_datatype_bytes(config.network_outputs[0].tvm_dtype_bits / 8), config_detail(config_mod)
{
  head_out_heatmap.resize(
    config.network_outputs[0].node_shape[1] * config.network_outputs[0].node_shape[2] *
    config.network_outputs[0].node_shape[3]);
  head_out_offset.resize(
    config.network_outputs[1].node_shape[1] * config.network_outputs[1].node_shape[2] *
    config.network_outputs[1].node_shape[3]);
  head_out_z.resize(
    config.network_outputs[2].node_shape[1] * config.network_outputs[2].node_shape[2] *
    config.network_outputs[2].node_shape[3]);
  head_out_dim.resize(
    config.network_outputs[3].node_shape[1] * config.network_outputs[3].node_shape[2] *
    config.network_outputs[3].node_shape[3]);
  head_out_rot.resize(
    config.network_outputs[4].node_shape[1] * config.network_outputs[4].node_shape[2] *
    config.network_outputs[4].node_shape[3]);
  head_out_vel.resize(
    config.network_outputs[5].node_shape[1] * config.network_outputs[5].node_shape[2] *
    config.network_outputs[5].node_shape[3]);
}

bool BackboneNeckHeadPostProcessor::schedule(const TVMArrayContainerVector & input)
{
  TVMArrayCopyToBytes(
    input[0].getArray(), head_out_heatmap.data(), head_out_heatmap.size() * output_datatype_bytes);
  TVMArrayCopyToBytes(
    input[1].getArray(), head_out_offset.data(), head_out_offset.size() * output_datatype_bytes);
  TVMArrayCopyToBytes(
    input[2].getArray(), head_out_z.data(), head_out_z.size() * output_datatype_bytes);
  TVMArrayCopyToBytes(
    input[3].getArray(), head_out_dim.data(), head_out_dim.size() * output_datatype_bytes);
  TVMArrayCopyToBytes(
    input[4].getArray(), head_out_rot.data(), head_out_rot.size() * output_datatype_bytes);
  TVMArrayCopyToBytes(
    input[5].getArray(), head_out_vel.data(), head_out_vel.size() * output_datatype_bytes);

  return true;
}

int main() {

  centerpoint::CenterPointConfig config(3, 4, 40000, {-89.6, -89.6, -3.0, 89.6, 89.6, 5.0}, 
    {0.32, 0.32, 8.0}, 1, 9, 0.35, 0.5, {0.3, 0.0, 0.3});
  const auto voxels_size =
    config.max_voxel_size_ * config.max_point_in_voxel_size_ * config.point_feature_size_;
  const auto coordinates_size = config.max_voxel_size_ * config.point_dim_size_;
    std::vector<int64_t> shape_coords{
    config_scatter.network_inputs[1].node_shape[0], config_scatter.network_inputs[1].node_shape[1]};
  TVMArrayContainer coords_tvm_ = TVMArrayContainer(
    shape_coords, config_scatter.network_inputs[1].tvm_dtype_code,
    config_scatter.network_inputs[1].tvm_dtype_bits,
    config_scatter.network_inputs[1].tvm_dtype_lanes, config_scatter.tvm_device_type,
    config_scatter.tvm_device_id);

  const auto encoder_in_features_size = config_en.network_inputs[0].node_shape[0] * config_en.network_inputs[0].node_shape[1]
    * config_en.network_inputs[0].node_shape[2];
  
  std::vector<float> encoder_in_features(encoder_in_features_size, 1.0);
  std::vector<int32_t> coords(coordinates_size, 0);
  for (size_t i=0; i < coordinates_size; i++)
  {
    coords[i] = rand() % 100;
  }

  TVMArrayCopyFromBytes(
    coords_tvm_.getArray(), coords.data(), coords.size() * sizeof(int32_t));
  // Voxel Encoder Pipeline.
  std::shared_ptr<VE_PrePT> VE_PreP = std::make_shared<VE_PrePT>(config_en, config);
  std::shared_ptr<IET> VE_IE = std::make_shared<IET>(config_en, "/home/xinyuwang/adehome/tvm_latest/tvm_example");

  // Backbone Neck Head Pipeline.
  std::shared_ptr<IET> BNH_IE = std::make_shared<IET>(config_bk, "/home/xinyuwang/adehome/tvm_latest/tvm_example");
  std::shared_ptr<BNH_PostPT> BNH_PostP = std::make_shared<BNH_PostPT>(config_bk, config);

  std::shared_ptr<TSE> scatter_ie = std::make_shared<TSE>(config_scatter, "/home/xinyuwang/adehome/tvm_latest/tvm_example", "scatter");
  std::shared_ptr<tvm_utility::pipeline::TowStagePipeline<VE_PrePT, IET, TSE, BNH_PostPT>>
    TSP_pipeline = std::make_shared<TSP>(VE_PreP, VE_IE, scatter_ie, BNH_IE, BNH_PostP);
  

  scatter_ie->set_coords(coords_tvm_);
  // MixedInputs voxel_inputs{num_voxels_, *voxels_, *num_points_per_voxel_, *coordinates_};
  auto bnh_output = TSP_pipeline->schedule(encoder_in_features);

  return 0;
}