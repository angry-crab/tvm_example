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

hardware_params = auto_scheduler.HardwareParams(
    max_shared_memory_per_block=49152,
    max_threads_per_block=1024,
    # The value `max_local_memory_per_block` is not used in AutoScheduler,
    # but is required by the API.
    max_local_memory_per_block=12345678,
    max_vthread_extent=8,
    warp_size=32,
    target=target,
)

network = "encoder"
log_file = "%s.json" % network

runner = auto_scheduler.LocalRunner(
        timeout=15,
        number=3,
        repeat=2,
        min_repeat_ms=100,
        cooldown_interval=0.0,
        enable_cpu_cache_flush=False,
        device=0,
)

tasks, task_weights = auto_scheduler.extract_tasks(
    mod["main"],
    params,
    target=target,
    hardware_params=hardware_params,
)
for idx, (task, task_weight) in enumerate(zip(tasks, task_weights)):
    print(
        f"==== Task {idx}: {task.desc} "
        f"(weight {task_weight} key: {task.workload_key}) ====="
    )
    print(task.compute_dag)

tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
tuner.tune(
    auto_scheduler.TuningOptions(
        num_measure_trials=3000,
        verbose=1,
        runner=runner,
        measure_callbacks=[
            auto_scheduler.RecordToFile(log_file),
        ],
    ),
    adaptive_training=False,
)

relay_build = {"graph": relay.build, "vm": relay.vm.compile}["graph"]
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(
        opt_level=3,
        config={"relay.backend.use_auto_scheduler": True},
    ):
        graph, lib, params = relay_build(mod, target=target, params=params)
        lib.export_library(output_model_path)

        with open(output_graph_path, 'w', encoding='utf-8') as graph_file:
            graph_file.write(graph)

        with open(output_param_path, 'wb') as param_file:
            param_file.write(relay.save_param_dict(params))