import onnx
import numpy as np
import onnxruntime as ort
import tvm
from tvm.contrib import graph_executor
import tvm.auto_scheduler as auto_scheduler
from tvm import relay, autotvm

yolox = "/home/xinyuwang/adehome/tvm_latest/tvm_example/model/yolox_nano.onnx"

onnx_ = onnx.load(yolox)
x = np.ones((1,3,416,416), dtype=np.float32)
input_name = "images"

ort_sess = ort.InferenceSession(onnx_.SerializeToString())
out_onnx = ort_sess.run(None, {input_name: x})

output_shape = (1,3549,85)

