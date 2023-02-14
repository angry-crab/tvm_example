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
output_model_path = "/home/xinyuwang/adehome/tvm_latest/tvm_example/deploy_lib.so"
output_graph_path = "/home/xinyuwang/adehome/tvm_latest/tvm_example/deploy_graph.json"
output_param_path = "/home/xinyuwang/adehome/tvm_latest/tvm_example/deploy_param.params"

onnx_ = onnx.load(model_encoder)
x = np.ones((40000,32,9), dtype=np.float32)
input_name = "input_features"

# onnx_ = onnx.load(model_head)
# x = np.ones((1,32,560,560), dtype=np.float32)
# input_name = "spatial_features"

target = "cuda"

shape_dict = {input_name: x.shape}

mod, params = relay.frontend.from_onnx(onnx_, shape_dict)

# print(tvm.transform.PassContext.list_configs())

# network = "encoder"
# log_file = "%s.log" % network
# dtype = "float32"

# tuning_option = {
#     "log_filename": log_file,
#     "tuner": "xgb",
#     "n_trial": 1000,
#     "early_stopping": 600,
#     "measure_option": autotvm.measure_option(
#         builder=autotvm.LocalBuilder(timeout=10),
#         runner=autotvm.LocalRunner(number=20, repeat=10, timeout=4, min_repeat_ms=150),
#     ),
# }

# tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# # Tune the extracted tasks sequentially.
# for i, task in enumerate(tasks):
#     prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
#     tuner_obj = XGBTuner(task, loss_type="rank")
#     tuner_obj.tune(
#         n_trial=min(tuning_option["n_trial"], len(task.config_space)),
#         early_stopping=tuning_option["early_stopping"],
#         measure_option=tuning_option["measure_option"],
#         callbacks=[
#             autotvm.callback.progress_bar(tuning_option["n_trial"], prefix=prefix),
#             autotvm.callback.log_to_file(tuning_option["log_filename"]),
#         ],
#     )

# prefix = "[Task %2d/%2d] " % (0 + 1, len(tasks))
# tuner_obj = XGBTuner(tasks[0], loss_type="rank")
# tuner_obj.tune(
#     n_trial=min(tuning_option["n_trial"], len(tasks[0].config_space)),
#     early_stopping=tuning_option["early_stopping"],
#     measure_option=tuning_option["measure_option"],
#     callbacks=[
#         autotvm.callback.progress_bar(tuning_option["n_trial"], prefix=prefix),
#         autotvm.callback.log_to_file(tuning_option["log_filename"]),
#     ],
# )

# with autotvm.apply_history_best(tuning_option["tuning_records"]):

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