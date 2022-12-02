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
    def main(pillar_features: T.Buffer[(40000, 1, 32), "float32"],
                coords: T.Buffer[(40000, 3), "int32"],
                spatial_features: T.Buffer[(1, 32, 560, 560), "float32"]):
        T.func_attr({"global_symbol": "scatter", "tir.noalias": True})
        for i, j in T.grid(40000, 32):
            with T.block("spatial_features"):
                vi, vj = T.axis.remap("SS", [i, j])
                # vi = T.axis.spatial(40000, i)
                # vj = T.axis.spatial(32, j)
                if(coords[vi, 0] >= 0):
                    spatial_features[0, vj, coords[vi, 1], coords[vi, 2]] = pillar_features[vi, 0, vj]

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

pillar_features = tvm.nd.array(np.random.rand(40000, 1, 32).astype("float32"))
coords = tvm.nd.array(np.random.randint(0, 200, (40000, 3), dtype="int32"))
spatial_features = tvm.nd.empty((1, 32, 560, 560), dtype="float32")
rt_lib = tvm.build(MyModule, target="llvm")
func = rt_lib["scatter"]
func(pillar_features, coords, spatial_features)
spatial_tvm = spatial_features.numpy()

spatial_python = scatter_py(pillar_features.numpy(), coords.numpy())

diff = np.absolute(spatial_tvm - spatial_python)
print(diff)
print(diff[np.where(diff > 0.001)])
print(len(diff[np.where(diff > 0.001)]))

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