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
                if(coords[vi, 0] > 0):
                    spatial_features[vj, coords[vi, 2], coords[vi, 1]] = pillar_features[vi, vj]

        # Y = T.alloc_buffer((128, 128), dtype="float32")
        # for i, j, k in T.grid(128, 128, 128):
        #     with T.block("Y"):
        #         vi = T.axis.spatial(128, i)
        #         vj = T.axis.spatial(128, j)
        #         vk = T.axis.reduce(128, k)
        #         with T.init():
        #             Y[vi, vj] = T.float32(0)
        #         Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        # for i, j in T.grid(128, 128):
        #     with T.block("C"):
        #         vi = T.axis.spatial(128, i)
        #         vj = T.axis.spatial(128, j)
        #         C[vi, vj] = T.max(Y[vi, vj], T.float32(0))

pillar_features = tvm.nd.array(0.5*np.ones((40000,32), dtype="float32"))
coords = tvm.nd.array(np.ones((40000,3), dtype="int32"))
spatial_features = tvm.nd.empty((32, 560, 560), dtype="float32")
rt_lib = tvm.build(MyModule, target="llvm")
func = rt_lib["main"]
func(pillar_features, coords, spatial_features)
spatial_tvm = spatial_features.numpy()
print(spatial_tvm)
# print(spatial_tvm[np.where(spatial_tvm > 0)])
print(len(spatial_tvm[np.where(spatial_tvm > 0)]))

def scatter_py(voxel_features, coords):

    nx = 560
    ny = 560

    # batch_canvas will be the final output.
    # Create the canvas for this sample
    canvas = np.zeros((32 *nx * ny),dtype="float32")

    # Only include non-empty pillars
    batch_mask = coords[:, 0] > 0

    this_coords = coords[batch_mask, :]
    indices = this_coords[:, 2] * nx + this_coords[:, 3]
    voxels = voxel_features[batch_mask, :]
    voxels = voxels.t()

    # Now scatter the blob back to the canvas.
    canvas[:, indices] = voxels

    # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)

    return canvas