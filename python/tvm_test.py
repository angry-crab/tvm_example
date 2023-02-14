import onnx
import numpy as np
import onnxruntime as ort
import tvm
from tvm.contrib import graph_executor
import tvm.auto_scheduler as auto_scheduler
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

model_encoder = "/home/xinyuwang/adehome/tvm_latest/tvm_example/model/pts_voxel_encoder_centerpoint.onnx"
model_head = "/home/xinyuwang/adehome/tvm_latest/tvm_example/model/pts_backbone_neck_head_centerpoint.onnx"
fcn = "/home/xinyuwang/adehome/tvm_latest/tvm_example/model/fcn-resnet50-12.onnx"

onnx_ = onnx.load(model_encoder)
x = np.ones((40000,32,9), dtype=np.float32)
input_name = "input_features"

ort_sess = ort.InferenceSession(onnx_.SerializeToString())
out_onnx = ort_sess.run(None, {input_name: x})

# onnx_ = onnx.load(model_head)
# x = np.ones((1,32,560,560), dtype=np.float32)
# input_name = "spatial_features"

# target = "cuda"
target = tvm.target.create('cuda')

shape_dict = {input_name: x.shape}

mod, params = relay.frontend.from_onnx(onnx_, shape_dict)

# print(mod.astext())

# print(type(mod))
# s = tvm.tir.Schedule(mod)

# s.work_on("main")
# s.get_block("tvmgen_default_fused_nn_dense")

# print(dir(s))

# vs = relay.analysis.all_vars(mod)
# print(vs)

# with tvm.transform.PassContext(opt_level=3, config={}):
#     lib = relay.build(mod, target=target, params=params)
#     print(lib.get_params())
    # source = lib.get_lib().imported_modules[0].get_source()
    # with open("cuda_code.txt", "a") as t:
    #     print(source, file=t)


with tvm.transform.PassContext(opt_level=3, config={}):
    lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))

    dtype = "float32"
    module.set_input(input_name, x)
    module.run()
    output_shape = (40000, 1, 32)
    # output_shape = (1, 5, 560, 560)
    out_idx = 0
    tvm_output = module.get_output(out_idx, tvm.nd.empty(output_shape)).numpy()

    print(tvm_output.shape)
    print(out_onnx[out_idx].shape)
    result = out_onnx[out_idx] - tvm_output
    print(result[np.where(result > 0.0001)])
    print(len(result[np.where(result > 0.0001)]))