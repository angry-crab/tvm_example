import onnx
import numpy as np
# import onnxruntime as ort
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
# import tvm.auto_scheduler as auto_scheduler
# from tvm.autotvm.tuner import XGBTuner
# from tvm import autotvm

model_encoder = "/home/xinyuwang/adehome/tvm_latest/tvm_example/pts_voxel_encoder_centerpoint.onnx"
model_head = "/home/xinyuwang/adehome/tvm_latest/tvm_example/pts_backbone_neck_head_centerpoint.onnx"
output_model_path = "/home/xinyuwang/adehome/tvm_latest/tvm_example/deploy_lib.so"
output_graph_path = "/home/xinyuwang/adehome/tvm_latest/tvm_example/deploy_graph.json"
output_param_path = "/home/xinyuwang/adehome/tvm_latest/tvm_example/deploy_param.params"

onnx_ = onnx.load(model_head)

# x = np.ones((40000,32,9), dtype=np.float32)
# input_name = "input_features"

x = np.ones((1,32,560,560), dtype=np.float32)
input_name = "spatial_features"

target = "llvm"

shape_dict = {input_name: x.shape}

mod, params = relay.frontend.from_onnx(onnx_, shape_dict)

# print(tvm.transform.PassContext.list_configs())

with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True, "relay.FuseOps.link_params": True}):
    graph, lib, params = relay.build(mod, target=target, params=params)

    # dev = tvm.device(str(target), 0)
    # module = graph_executor.GraphModule(lib["default"](dev))

    # dtype = "float32"
    # module.set_input(input_name, x)
    # module.run()
    # output_shape = (1,12,864,864)
    # tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
    # print(tvm_output.shape)
    lib.export_library(output_model_path)

    with open(output_graph_path, 'w', encoding='utf-8') as graph_file:
        graph_file.write(graph)

    with open(output_param_path, 'wb') as param_file:
        param_file.write(relay.save_param_dict(params))