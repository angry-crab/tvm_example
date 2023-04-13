// Copyright 2022 AutoCore Ltd., TIER IV, Inc.
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

#ifndef LIDAR_CENTERPOINT_TVM__UTILS_HPP_
#define LIDAR_CENTERPOINT_TVM__UTILS_HPP_

#include <cstddef>
#include <cstdint>
#include <cmath>

namespace centerpoint
{

struct Box3D
{
  // initializer not allowed for __shared__ variable
  int32_t label;
  float score;
  float x;
  float y;
  float z;
  float length;
  float width;
  float height;
  float yaw;
  float vel_x;
  float vel_y;

  int operator-(Box3D& a)
  {
    int issue = 0;
    if(label != a.label)
      issue++;
    if(std::fabs(score - a.score) > 0.001)
      issue++;
    if(std::fabs(x - a.x) > 0.001)
      issue++;
    if(std::fabs(y - a.y) > 0.001)
      issue++;
    if(std::fabs(z - a.z) > 0.001)
      issue++;
    if(std::fabs(length - a.length) > 0.001)
      issue++;
    if(std::fabs(width - a.width) > 0.001)
      issue++;
    if(std::fabs(height - a.height) > 0.001)
      issue++;
    if(std::fabs(yaw - a.yaw) > 0.001)
      issue++;
    if(std::fabs(vel_x - a.vel_x) > 0.001)
      issue++;
    if(std::fabs(vel_y - a.vel_y) > 0.001)
      issue++;
    return issue;
  }
};

// cspell: ignore divup
std::size_t divup(const std::size_t a, const std::size_t b)
{
  if (a == 0) {
    throw std::runtime_error("A dividend of divup isn't positive.");
  }
  if (b == 0) {
    throw std::runtime_error("A divisor of divup isn't positive.");
  }

  return (a + b - 1) / b;
}
}  // namespace lidar_centerpoint_tvm

#endif  // LIDAR_CENTERPOINT_TVM__UTILS_HPP_
