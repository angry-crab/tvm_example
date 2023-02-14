#include "network_trt.hpp"
#include "cuda_utils.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <chrono>

int main()
{
    std::string encoder_engine_path = "/home/xinyuwang/adehome/tvm_latest/tvm_example/model/pts_voxel_encoder_centerpoint.engine";
    std::string encoder_onnx_path = "/home/xinyuwang/adehome/tvm_latest/tvm_example/model/pts_voxel_encoder_centerpoint.onnx";
    std::string header_engine_path = "/home/xinyuwang/adehome/tvm_latest/tvm_example/model/pts_backbone_neck_head_centerpoint.engine";
    std::string header_onnx_path = "/home/xinyuwang/adehome/tvm_latest/tvm_example/model/pts_backbone_neck_head_centerpoint.onnx";
    centerpoint::CenterPointConfig config(3, 4, 40000, {-89.6, -89.6, -3.0, 89.6, 89.6, 5.0}, 
        {0.32, 0.32, 8.0}, 1, 9, 0.35, 0.5, {0.3, 0.0, 0.3});
    centerpoint::VoxelEncoderTRT encoder(config, false);
    encoder.init(encoder_onnx_path, encoder_engine_path, "fp32");
    encoder.context_->setBindingDimensions(0, nvinfer1::Dims3(
        config.max_voxel_size_, config.max_point_in_voxel_size_, config.encoder_in_feature_size_));
    
    std::vector<std::size_t> out_channel_sizes = {
    config.class_size_,        config.head_out_offset_size_, config.head_out_z_size_,
    config.head_out_dim_size_, config.head_out_rot_size_,    config.head_out_vel_size_};
    centerpoint::HeadTRT header(out_channel_sizes, config, false);
    header.init(header_onnx_path, header_engine_path, "fp32");
    header.context_->setBindingDimensions(
        0, nvinfer1::Dims4(
            config.batch_size_, config.encoder_out_feature_size_, config.grid_size_y_,
            config.grid_size_x_));
    
    const auto voxels_size =
    config.max_voxel_size_ * config.max_point_in_voxel_size_ * config.point_feature_size_;
    const auto coordinates_size = config.max_voxel_size_ * config.point_dim_size_;
    const auto encoder_in_feature_size_ =
    config.max_voxel_size_ * config.max_point_in_voxel_size_ * config.encoder_in_feature_size_;
    const auto pillar_features_size = config.max_voxel_size_ * config.encoder_out_feature_size_;
    const auto spatial_features_size_ =
    config.grid_size_x_ * config.grid_size_y_ * config.encoder_out_feature_size_;
    const auto grid_xy_size = config.down_grid_size_x_ * config.down_grid_size_y_;

    cuda::unique_ptr<float[]> encoder_in_features_d_ = cuda::make_unique<float[]>(encoder_in_feature_size_);
    cuda::unique_ptr<float[]> pillar_features_d_ = cuda::make_unique<float[]>(pillar_features_size);
    cuda::unique_ptr<float[]> spatial_features_d_ = cuda::make_unique<float[]>(spatial_features_size_);
    cuda::unique_ptr<float[]> head_out_heatmap_d_ = cuda::make_unique<float[]>(grid_xy_size * config.class_size_);
    cuda::unique_ptr<float[]> head_out_offset_d_ = cuda::make_unique<float[]>(grid_xy_size * config.head_out_offset_size_);
    cuda::unique_ptr<float[]> head_out_z_d_ = cuda::make_unique<float[]>(grid_xy_size * config.head_out_z_size_);
    cuda::unique_ptr<float[]> head_out_dim_d_ = cuda::make_unique<float[]>(grid_xy_size * config.head_out_dim_size_);
    cuda::unique_ptr<float[]> head_out_rot_d_ = cuda::make_unique<float[]>(grid_xy_size * config.head_out_rot_size_);
    cuda::unique_ptr<float[]> head_out_vel_d_ = cuda::make_unique<float[]>(grid_xy_size * config.head_out_vel_size_);

    cudaStream_t stream_{nullptr};
    cudaStreamCreate(&stream_);

    std::vector<float> encoder_in_features_host(encoder_in_feature_size_, 0);
    // std::fill(encoder_in_features_host.begin(), encoder_in_features_host.end(), 1.0);

    CHECK_CUDA_ERROR(cudaMemsetAsync(
        encoder_in_features_d_.get(), 0, encoder_in_feature_size_ * sizeof(float), stream_));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        encoder_in_features_d_.get(), encoder_in_features_host.data(), encoder_in_feature_size_ * sizeof(float),
        cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

    std::vector<float> spatial_host(spatial_features_size_, 0);

    CHECK_CUDA_ERROR(cudaMemsetAsync(
        spatial_features_d_.get(), 0, spatial_features_size_ * sizeof(float), stream_));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        spatial_features_d_.get(), spatial_host.data(), spatial_features_size_ * sizeof(float),
        cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

    std::chrono::duration<long int, std::ratio<1, 1000>> d(0);
    for (int i = 0; i < 100; i++)
    {
        auto start = std::chrono::system_clock::now();
    
        std::vector<void *> encoder_buffers{encoder_in_features_d_.get(), pillar_features_d_.get()};
        encoder.context_->enqueueV2(encoder_buffers.data(), stream_, nullptr);

        std::vector<void *> head_buffers = {spatial_features_d_.get(), head_out_heatmap_d_.get(),
                                        head_out_offset_d_.get(),  head_out_z_d_.get(),
                                        head_out_dim_d_.get(),     head_out_rot_d_.get(),
                                        head_out_vel_d_.get()};
        header.context_->enqueueV2(head_buffers.data(), stream_, nullptr);


        auto stop = std::chrono::system_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        d += duration;
        // std::cout << "run time: " << duration.count() << std::endl;
    }

    std::cout << "run time: " << d.count() / 100 << std::endl;

    // std::vector<float> pillar_features_host(pillar_features_size, 0);
    // CHECK_CUDA_ERROR(cudaMemcpyAsync(
    //     pillar_features_host.data(), pillar_features_d_.get(), pillar_features_size * sizeof(float),
    //     cudaMemcpyDeviceToHost));
    // CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

    // std::size_t count = 0;
    // for (std::size_t i; i < pillar_features_size; i++)
    // {
    //     if(pillar_features_host[i] != 0)
    //     {
    //         count++;
    //     }
    // }

    // std::cout << "None Zero : " << count << std::endl;

    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }

    return 0;
}