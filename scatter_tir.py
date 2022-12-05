import numpy as np

import tvm
from tvm.script import tir as T
import tvm.relay as relay
from tvm.contrib import graph_executor

import onnx
# from tvm.ir.module import IRModule
# from tvm.contrib import graph_executor
# from tvm import relay

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def scatter(pillar_features: T.Buffer[(40000, 1, 32), "float32"],
                coords: T.Buffer[(40000, 3), "int32"],
                spatial_features: T.Buffer[(1, 32, 560, 560), "float32"]):
        # coord = T.alloc_buffer([3], "int32")
        T.func_attr({"global_symbol": "scatter", "tir.noalias": True})
        # for i, j, k in T.grid(32, 560, 560):
        #     with T.block("spatial_features"):
        #         vi, vj, vk = T.axis.remap("SSS", [i, j, k])
        #         spatial_features[0, vi, vj, vk] = T.float32(0)
        for i, j in T.grid(40000, 32):
            with T.block("spatial_features"):
                vi, vj = T.axis.remap("SS", [i, j])
                if(coords[vi,0] >= 0):
                    spatial_features[0, vj, coords[vi,1], coords[vi,2]] = pillar_features[vi, 0, vj]

@tvm.script.ir_module
class TestModule:
    @T.prim_func
    def scatter(X: T.Buffer[(1, 2), "float32"],
                coords: T.Buffer[(1, 2), "int32"],
                Y: T.Buffer[(2, 6), "float32"]):
        T.func_attr({"global_symbol": "scatter", "tir.noalias": True})
        for i, j in T.grid(1, 2):
            with T.block("Y"):
                vi, vj = T.axis.remap("SS", [i, j])
                # vi = T.axis.spatial(40000, i)
                # vj = T.axis.spatial(32, j)
                # T.reads(X[vi, vj])
                # T.writes(Y[vi,0])
                # with T.init():
                #     Y[vi,vj] = T.float32(0)
                if(X[vi,vj] > 0.0):
                    Y[1,coords[0,vj]] = X[vi,vj]

def scatter_py(voxel_features, coords):
    nx = 560
    ny = 560
    # Create the canvas for this sample
    canvas = np.zeros((32,nx * ny),dtype="float32")
    # Only include non-empty pillars
    batch_mask = coords[:, 0] > 0
    this_coords = coords[batch_mask, :]
    indices = this_coords[:, 1] * nx + this_coords[:, 2]
    voxels = voxel_features[batch_mask, 0, :]
    voxels = voxels.T
    # Now scatter the blob back to the canvas.
    canvas[:, indices] = voxels
    # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
    return canvas.reshape(1,32,560,560)

# xx = np.ones((1, 2)).astype("float32")
# xx[0,1] = 20.0
# co = np.zeros((1, 2)).astype("int32")
# co[0,0] = 3
# co[0,1] = 5
# print(xx)
# yy = np.zeros((2, 6)).astype("float32")
# X = tvm.nd.array(xx)
# Y = tvm.nd.array(yy)
# C = tvm.nd.array(co)
# rt_lib = tvm.build(TestModule, target="llvm")
# func = rt_lib["scatter"]
# func(X, C, Y)
# print(Y.numpy())

# pillar_features = tvm.nd.array(np.random.rand(40000, 1, 32).astype("float32"))
# coords = tvm.nd.array(np.random.randint(0, 200, (40000, 3), dtype="int32"))
# spatial_features = tvm.nd.empty((1, 32, 560, 560), dtype="float32")
rt_lib = tvm.build(MyModule, target="llvm")
func = rt_lib["scatter"]
# func(pillar_features, coords, spatial_features)
# spatial_tvm = spatial_features.numpy()

# spatial_python = scatter_py(pillar_features.numpy(), coords.numpy())

# diff = np.absolute(spatial_tvm - spatial_python)
# print(diff)
# print(diff[np.where(diff > 0.001)])
# print(len(diff[np.where(diff > 0.001)]))

rt_lib.export_library("scatter.so")


# model_encoder = "/home/xinyuwang/adehome/tvm_latest/tvm_example/pts_voxel_encoder_centerpoint.onnx"
# onnx_encoder = onnx.load(model_encoder)
# target = "llvm"
# input_name = "spatial_features"
# x = np.ones((40000,32,9), dtype=np.float32)
# shape_dict = {input_name: x.shape}
# mod, params = relay.frontend.from_onnx(onnx_encoder, shape_dict)
# mod.update(MyModule)
# with tvm.transform.PassContext(opt_level=3, config={}):
#     lib = relay.build(mod, target=target, params=params)
#     dev = tvm.device(str(target), 0)
#     module = graph_executor.GraphModule(lib["default"](dev))
#     dtype = "float32"
#     module.set_input(input_name, x)
#     module.run()
#     output_shape = (40000, 1, 32)
#     out_idx = 0
#     tvm_output = module.get_output(out_idx, tvm.nd.empty(output_shape)).numpy()