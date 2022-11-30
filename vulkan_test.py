import onnx
import numpy as np
import onnxruntime as ort
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm

fcn = "/home/xinyuwang/adehome/tvm_test/tinyyolov2-8.onnx"

onnx_fcn = onnx.load(fcn)

x = np.ones((1,3,416,416), dtype=np.float32)

ort_sess = ort.InferenceSession(onnx_fcn.SerializeToString())
out_onnx = ort_sess.run(None, {'image': x})

target = "opencl"

input_name = "image"
shape_dict = {input_name: x.shape}

mod, params = relay.frontend.from_onnx(onnx_fcn, shape_dict)

with tvm.transform.PassContext(opt_level=3, config={}):
    lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))

    dtype = "float32"
    module.set_input(input_name, x)
    module.run()
    output_shape = (1,125,13,13)
    tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
    print(tvm_output.shape)
    print(out_onnx[0].shape)
    result = out_onnx[0] - tvm_output
    print(result[np.where(result > 0.0001)])
    print(len(result[np.where(result > 0.0001)]))