import onnx
import numpy as np
# import onnxruntime as ort
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
# import tvm.auto_scheduler as auto_scheduler
# from tvm.autotvm.tuner import XGBTuner
# from tvm import autotvm

bcnn = "/home/xinyuwang/adehome/tvm_latest/tvm_example/bcnn.onnx"
output_model_path = "/home/xinyuwang/adehome/tvm_latest/tvm_example/deploy_lib.so"
output_graph_path = "/home/xinyuwang/adehome/tvm_latest/tvm_example/deploy_graph.json"
output_param_path = "/home/xinyuwang/adehome/tvm_latest/tvm_example/deploy_param.params"

onnx_bcnn = onnx.load(bcnn)

x = np.ones((1,4,864,864), dtype=np.float32)

# ort_sess = ort.InferenceSession(onnx_bcnn.SerializeToString())
# out_onnx = ort_sess.run(None, {'image': x})

target = "cuda"

input_name = "data"
shape_dict = {input_name: x.shape}

mod, params = relay.frontend.from_onnx(onnx_bcnn, shape_dict)

with tvm.transform.PassContext(opt_level=3, config={}):
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