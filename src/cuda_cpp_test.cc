#include "cuda_utils.hpp"
#include "postprocess_kernel.hpp"
#include "generate_detected_boxes.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <chrono>

int main()
{
  centerpoint::CenterPointConfig config(3, 4, 40000, {-89.6, -89.6, -3.0, 89.6, 89.6, 5.0}, 
      {0.32, 0.32, 8.0}, 1, 9, 0.35, 0.5, {0.3, 0.3, 0.3, 0.3, 0.0});
  // centerpoint_tiny
  // centerpoint::CenterPointConfig config(3, 4, 40000, {-76.8, -76.8, -2.0, 76.8, 76.8, 4.0}, 
  //     {0.32, 0.32, 6.0}, 2, 9, 0.35, 0.5, {0.3, 0.0, 0.3});

  std::vector<std::size_t> out_channel_sizes = {
  config.class_size_,        config.head_out_offset_size_, config.head_out_z_size_,
  config.head_out_dim_size_, config.head_out_rot_size_,    config.head_out_vel_size_};
  
  const auto voxels_size =
  config.max_voxel_size_ * config.max_point_in_voxel_size_ * config.point_feature_size_;
  const auto coordinates_size = config.max_voxel_size_ * config.point_dim_size_;
  const auto grid_xy_size = config.down_grid_size_x_ * config.down_grid_size_y_;

  cuda::unique_ptr<float[]> head_out_heatmap_d_ = cuda::make_unique<float[]>(grid_xy_size * config.class_size_);
  cuda::unique_ptr<float[]> head_out_offset_d_ = cuda::make_unique<float[]>(grid_xy_size * config.head_out_offset_size_);
  cuda::unique_ptr<float[]> head_out_z_d_ = cuda::make_unique<float[]>(grid_xy_size * config.head_out_z_size_);
  cuda::unique_ptr<float[]> head_out_dim_d_ = cuda::make_unique<float[]>(grid_xy_size * config.head_out_dim_size_);
  cuda::unique_ptr<float[]> head_out_rot_d_ = cuda::make_unique<float[]>(grid_xy_size * config.head_out_rot_size_);
  cuda::unique_ptr<float[]> head_out_vel_d_ = cuda::make_unique<float[]>(grid_xy_size * config.head_out_vel_size_);

  cudaStream_t stream_{nullptr};
  cudaStreamCreate(&stream_);

  
  return 0;
}