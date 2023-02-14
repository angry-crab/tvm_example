#include <pipeline.hpp>
#include <centerpoint_config.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

// namespace centerpoint
// {
using tvm_utility::pipeline::TVMArrayContainer;
using tvm_utility::pipeline::TVMArrayContainerVector;

struct MixedInputs
{
  // The number of non-empty voxel
  std::size_t num_voxels;
  // The voxel features (point info in each voxel) or pillar features.
  std::vector<float> features;
  // The number of points in each voxel.
  std::vector<float> num_points_per_voxel;
  // The index from voxel number to its 3D position
  std::vector<int32_t> coords;
};

class TVMScatterIE : public tvm_utility::pipeline::InferenceEngine
{
public:
  explicit TVMScatterIE(
    tvm_utility::pipeline::InferenceEngineTVMConfig config, const std::string & pkg_name,
    const std::string & function_name);
  TVMArrayContainerVector schedule(const TVMArrayContainerVector & input);
  void set_coords(TVMArrayContainer coords) { coords_ = coords; }

  tvm_utility::pipeline::InferenceEngineTVMConfig config_;
  TVMArrayContainer coords_;
  TVMArrayContainerVector output_;
  tvm::runtime::PackedFunc scatter_function;
};

class VoxelEncoderPreProcessor
: public tvm_utility::pipeline::PreProcessor<std::vector<float>>
{
public:
  /// \brief Constructor.
  /// \param[in] config The TVM configuration.
  /// \param[in] config_mod The centerpoint model configuration.
  explicit VoxelEncoderPreProcessor(
    const tvm_utility::pipeline::InferenceEngineTVMConfig & config,
    const centerpoint::CenterPointConfig & config_mod);

  /// \brief Convert the voxel_features to encoder_in_features (a TVM array).
  /// \param[in] voxel_inputs The voxel features related input
  /// \return A TVM array containing the encoder_in_features.
  /// \throw std::runtime_error If the features are incorrectly configured.
  TVMArrayContainerVector schedule(const std::vector<float> & voxel_inputs);

  const int64_t max_voxel_size;
  const int64_t max_point_in_voxel_size;
  const int64_t encoder_in_feature_size;
  const int64_t input_datatype_bytes;
  const centerpoint::CenterPointConfig config_detail;
  std::vector<float> encoder_in_features;
  TVMArrayContainer output;
};

class BackboneNeckHeadPostProcessor
: public tvm_utility::pipeline::PostProcessor<bool>
{
public:
  /// \brief Constructor.
  /// \param[in] config The TVM configuration.
  /// \param[in] config_mod The centerpoint model configuration.
  explicit BackboneNeckHeadPostProcessor(
    const tvm_utility::pipeline::InferenceEngineTVMConfig & config,
    const centerpoint::CenterPointConfig & config_mod);

  /// \brief Copy the inference result.
  /// \param[in] input The result of the inference engine.
  /// \return The inferred data.
  bool schedule(const TVMArrayContainerVector & input);

  const int64_t output_datatype_bytes;
  const centerpoint::CenterPointConfig config_detail;
  std::vector<float> head_out_heatmap;
  std::vector<float> head_out_offset;
  std::vector<float> head_out_z;
  std::vector<float> head_out_dim;
  std::vector<float> head_out_rot;
  std::vector<float> head_out_vel;
};
// }