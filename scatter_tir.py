import tvm
import numpy as np
from tvm.script import tir as T
# from tvm.ir.module import IRModule
# from tvm.contrib import graph_executor
# from tvm import relay

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def scatter(pillar_features: T.Buffer[(40000, 32), "float32"],
                coords: T.Buffer[(40000, 3), "int32"],
                spatial_features: T.Buffer[(32, 560, 560), "float32"]):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j in T.grid(40000, 32):
            with T.block("spatial_features"):
                vi, vj = T.axis.remap("SS", [i, j])
                # vi = T.axis.spatial(40000, i)
                # vj = T.axis.spatial(32, j)
                if(coords[vi, 0] >= 0):
                    spatial_features[vj, coords[vi, 1], coords[vi, 2]] = pillar_features[vi, vj]

def scatter_py(voxel_features, coords):
    nx = 560
    ny = 560
    # batch_canvas will be the final output.
    # Create the canvas for this sample
    canvas = np.zeros((32,nx * ny),dtype="float32")
    # Only include non-empty pillars
    batch_mask = coords[:, 0] > 0
    this_coords = coords[batch_mask, :]
    indices = this_coords[:, 1] * nx + this_coords[:, 2]
    voxels = voxel_features[batch_mask, :]
    voxels = voxels.T
    # Now scatter the blob back to the canvas.
    canvas[:, indices] = voxels
    # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
    return canvas.reshape(32,560,560)

pillar_features = tvm.nd.array(np.random.rand(40000,32).astype("float32"))
coords = tvm.nd.array(np.random.randint(0, 200, (40000,3), dtype="int32"))
spatial_features = tvm.nd.empty((32, 560, 560), dtype="float32")
rt_lib = tvm.build(MyModule, target="llvm")
func = rt_lib["main"]
func(pillar_features, coords, spatial_features)
spatial_tvm = spatial_features.numpy()
# print(spatial_tvm)
# print(spatial_tvm[np.where(spatial_tvm > 0)])
# print(len(spatial_tvm[np.where(spatial_tvm > 0)]))

spatial_python = scatter_py(pillar_features.numpy(), coords.numpy())

# print(coords.numpy())
diff = np.absolute(spatial_tvm - spatial_python)
print(diff)
print(diff[np.where(diff > 0.001)])
print(len(diff[np.where(diff > 0.001)]))