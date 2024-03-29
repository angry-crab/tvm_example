#include "cuda_utils.hpp"
#include "postprocess_kernel.hpp"
#include "generate_detected_boxes.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <random>

float randFloat()
{
    return (float)(rand()) / (float)(rand());
}

float randFloatUnit()
{
    return (float)(rand()) / (float)(RAND_MAX);
}

using namespace centerpoint;

int main()
{
  // test
  centerpoint::CenterPointConfig config(3, 4, 100, {-10, -10, -1.0, 10, 10, 1.0}, 
    {1, 1, 1.0}, 1, 9, 0.35, 0.5, {0.3, 0.3, 0.3, 0.3, 0.0});
  // centerpoint
  // centerpoint::CenterPointConfig config(3, 4, 40000, {-89.6, -89.6, -3.0, 89.6, 89.6, 5.0}, 
  //     {0.32, 0.32, 8.0}, 1, 9, 0.35, 0.5, {0.3, 0.3, 0.3, 0.3, 0.0});
  // centerpoint_tiny
  // centerpoint::CenterPointConfig config(3, 4, 40000, {-76.8, -76.8, -2.0, 76.8, 76.8, 4.0}, 
  //     {0.32, 0.32, 6.0}, 2, 9, 0.35, 0.5, {0.3, 0.0, 0.3});

  PostProcessCUDA p(config);

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

  std::vector<float> heatmap(grid_xy_size * config.class_size_, 0);
  std::vector<float> offset(grid_xy_size * config.head_out_offset_size_, 0);
  std::vector<float> z(grid_xy_size * config.head_out_z_size_, 0);
  std::vector<float> dim(grid_xy_size * config.head_out_dim_size_, 0);
  std::vector<float> rot(grid_xy_size * config.head_out_rot_size_, 0);
  std::vector<float> vel(grid_xy_size * config.head_out_vel_size_, 0);

  std::srand(static_cast <unsigned>(std::time(0)));

  // for(int i = 0; i< 100; i++)
  //   heatmap[i] = 1.5;
  // for(int i = 0; i< 100; i++)
  //   offset[i] = 2.5;
  // for(int i = 0; i< 100; i++)
  //   z[i] = 3.5;
  // for(int i = 0; i< 100; i++)
  //   dim[i] = 4.5;
  // for(int i = 0; i< 100; i++)
  //   rot[i] = 5.5;
  // for(int i = 0; i< 100; i++)
  //   vel[i] = 6.5;

  for(int i = 0; i< heatmap.size(); i++)
    heatmap[i] = randFloat();
  for(int i = 0; i< offset.size(); i++)
    offset[i] = randFloat();
  for(int i = 0; i< z.size(); i++)
    z[i] = randFloat();
  for(int i = 0; i< dim.size(); i++)
    dim[i] = randFloat();
  for(int i = 0; i< rot.size(); i++)
    rot[i] = randFloat();
  for(int i = 0; i< vel.size(); i++)
    vel[i] = randFloat();

  dim[0] = 0.0;
  // for(int i = 1; i< heatmap.size(); i++)
  //   heatmap[i] = heatmap[i-1] + 0.1;
  // for(int i = 1; i< offset.size(); i++)
  //   offset[i] = offset[i-1] + 0.1;
  // for(int i = 1; i< z.size(); i++)
  //   z[i] = z[i-1] + 0.1;
  for(int i = 1; i< dim.size(); i++)
    dim[i] = dim[i-1] + 0.01;
  // for(int i = 1; i< rot.size(); i++)
  //   rot[i] = rot[i-1] + 0.1;
  // for(int i = 1; i< vel.size(); i++)
  //   vel[i] = vel[i-1] + 0.1;

  CHECK_CUDA_ERROR(cudaMemsetAsync(
    head_out_heatmap_d_.get(), 0, heatmap.size() * sizeof(float), stream_));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    head_out_heatmap_d_.get(), heatmap.data(), heatmap.size() * sizeof(float),
    cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  CHECK_CUDA_ERROR(cudaMemsetAsync(
    head_out_offset_d_.get(), 0, offset.size() * sizeof(float), stream_));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    head_out_offset_d_.get(), offset.data(), offset.size() * sizeof(float),
    cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  CHECK_CUDA_ERROR(cudaMemsetAsync(
    head_out_z_d_.get(), 0, z.size() * sizeof(float), stream_));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    head_out_z_d_.get(), z.data(), z.size() * sizeof(float),
    cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  CHECK_CUDA_ERROR(cudaMemsetAsync(
    head_out_dim_d_.get(), 0, dim.size() * sizeof(float), stream_));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    head_out_dim_d_.get(), dim.data(), dim.size() * sizeof(float),
    cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  CHECK_CUDA_ERROR(cudaMemsetAsync(
    head_out_rot_d_.get(), 0, rot.size() * sizeof(float), stream_));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    head_out_rot_d_.get(), rot.data(), rot.size() * sizeof(float),
    cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  CHECK_CUDA_ERROR(cudaMemsetAsync(
    head_out_vel_d_.get(), 0, vel.size() * sizeof(float), stream_));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    head_out_vel_d_.get(), vel.data(), vel.size() * sizeof(float),
    cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  std::vector<Box3D> det_boxes3d_d_;
  CHECK_CUDA_ERROR(p.generateDetectedBoxes3D_launch(
    head_out_heatmap_d_.get(), head_out_offset_d_.get(), head_out_z_d_.get(), head_out_dim_d_.get(),
    head_out_rot_d_.get(), head_out_vel_d_.get(), det_boxes3d_d_, stream_));

  std::vector<Box3D> det_boxes3d;
  generateDetectedBoxes3D(heatmap, offset, z, dim, rot, vel, config, det_boxes3d);

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    heatmap.data(), head_out_heatmap_d_.get(), heatmap.size() * sizeof(float),
    cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    offset.data(), head_out_offset_d_.get(), offset.size() * sizeof(float),
    cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    z.data(), head_out_z_d_.get(), z.size() * sizeof(float),
    cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    dim.data(), head_out_dim_d_.get(), dim.size() * sizeof(float),
    cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    rot.data(), head_out_rot_d_.get(), rot.size() * sizeof(float),
    cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    vel.data(), head_out_vel_d_.get(), vel.size() * sizeof(float),
    cudaMemcpyDeviceToHost));
  
  int count = 0;
  std::vector<std::vector<float>> wrong;
  std::vector<int> wrong_idx;
  for(int i = 0; i < det_boxes3d_d_.size(); i++)
  {
    std::vector<float> ret = det_boxes3d_d_[i] - det_boxes3d[i];
    bool save = false;
    for (auto i : ret)
    {
      if (i>0)
        save = true;
    }
    if(save)
    {
      count++;
      wrong_idx.push_back(i);
    }
    wrong.push_back(ret);
  }
  std::cerr <<  "total : " << det_boxes3d_d_.size() <<" wrong : " << count << std::endl;
  // std::cerr << "wrong idx : " << wrong_idx[0] << " " << wrong_idx[1] << " " << wrong_idx[wrong_idx.size()-1] << std::endl; 
  std::cerr << "? : " << std::endl;
  // for (int i=0; i<wrong_idx.size(); i++)
  for (int i=0; i<4; i++)
  {
    std::cerr << wrong_idx[i] << std::endl;
    for (auto i : wrong[wrong_idx[i]])
    {
      std::cerr << " " << i;
    }
    std::cerr << std::endl;
  }
  std::cerr << std::endl;
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }
  return 0;
}