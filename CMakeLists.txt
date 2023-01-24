# Note: This is meant to be built in autoware docker images

# TODO: Set up a autoware free environment

cmake_minimum_required(VERSION 3.2)
project(tvm_cpp)

find_package(tvm_vendor CONFIG REQUIRED)
set(tvm_runtime_DIR ${tvm_vendor_DIR})
find_package(tvm_runtime CONFIG REQUIRED)
message(WARNING "${tvm_runtime_LIBRARIES}")

add_executable(tvm_cpp_test
        src/cpp_test_tvm.cc
)
target_link_libraries(tvm_cpp_test
        "${tvm_runtime_LIBRARIES}"
)
target_include_directories(tvm_cpp_test SYSTEM PUBLIC
        "include"
        "${tvm_vendor_INCLUDE_DIRS}"
)
