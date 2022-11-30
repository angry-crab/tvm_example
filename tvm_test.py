import onnx
import numpy as np
import onnxruntime as ort
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm

model_encoder = "/home/xinyuwang/adehome/tvm_test/pts_voxel_encoder_centerpoint.onnx"
model_head = "/home/xinyuwang/adehome/tvm_test/pts_backbone_neck_head_centerpoint.onnx"
fcn = "/home/xinyuwang/adehome/tvm_test/fcn-resnet50-12.onnx"
onnx_encoder = onnx.load(model_encoder)
onnx_head = onnx.load(model_head)
# onnx_fcn = onnx.load(fcn)

# x = np.ones((40000,32,9), dtype=np.float32)
x = np.zeros((1,32,560,560), dtype=np.float32)

ort_sess = ort.InferenceSession(onnx_head.SerializeToString())
# out_onnx = ort_sess.run(None, {'input_features': x})
out_onnx = ort_sess.run(None, {'spatial_features': x})

# print(out_onnx.shape)
# np.savez("x", data=x)

# mm = tvm.runtime.load_module("/home/xinyuwang/adehome/tvm_test/mod.tar")

# out_lib = np.load("/home/xinyuwang/adehome/tvm_test/lib_llvm.npz")

# print(out_lib.files)

# target = tvm.target.cuda(model="3070ti",arch="sm_86")
target = "llvm"

input_name = "spatial_features"
shape_dict = {input_name: x.shape}

mod, params = relay.frontend.from_onnx(onnx_head, shape_dict)

# mod.show()

# number = 20
# repeat = 3
# min_repeat_ms = 250  # since we're tuning on a CPU, can be set to 0
# timeout = 10  # in seconds

# # create a TVM runner
# runner = autotvm.LocalRunner(
#     number=number,
#     repeat=repeat,
#     timeout=timeout,
#     min_repeat_ms=min_repeat_ms,
#     enable_cpu_cache_flush=True,
# )

# tuning_option = {
#     "tuner": "xgb",
#     "trials": 9000,
#     "early_stopping": 600,
#     "measure_option": autotvm.measure_option(
#         builder=autotvm.LocalBuilder(timeout=15), runner=runner
#     ),
#     "tuning_records": "encoder-autotuning.json",
# }

# tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# # Tune the extracted tasks sequentially.
# for i, task in enumerate(tasks):
#     prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
#     tuner_obj = XGBTuner(task, loss_type="rank")
#     tuner_obj.tune(
#         n_trial=min(tuning_option["trials"], len(task.config_space)),
#         early_stopping=tuning_option["early_stopping"],
#         measure_option=tuning_option["measure_option"],
#         callbacks=[
#             autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
#             autotvm.callback.log_to_file(tuning_option["tuning_records"]),
#         ],
#     )


# with autotvm.apply_history_best(tuning_option["tuning_records"]):
with tvm.transform.PassContext(opt_level=3, config={}):
    lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))

    dtype = "float32"
    module.set_input(input_name, x)
    module.run()
    # output_shape = (40000, 1, 32)
    output_shape = (1, 3, 560, 560)
    out_idx = 0
    tvm_output = module.get_output(out_idx, tvm.nd.empty(output_shape)).numpy()

    print(tvm_output.shape)
    print(out_onnx[out_idx].shape)
    # for i in range(len(out_onnx)):
    # idx = "output_" + str(0)
    result = out_onnx[out_idx] - tvm_output
    print(result[np.where(result > 0.0001)])
    print(len(result[np.where(result > 0.0001)]))