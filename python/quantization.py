import onnx
import numpy as np
import onnxruntime as ort
import cv2
import glob

import tvm
from tvm.contrib import graph_executor
import tvm.auto_scheduler as auto_scheduler
from tvm import relay, autotvm

yolox = "/home/xinyuwang/adehome/tvm_example/model/yolox_nano.onnx"

def download_coco():
    import fiftyone as fo
    import fiftyone.zoo as foz
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="test",
        label_types=["detections"],
        max_samples=10000,
    )
    sample = dataset.first()

class Dataset:
    def __init__(self, path):
        self.path = path
        self.paths = []
        for i in glob.glob(path+'/*.jpg'):
            self.paths.append(i)
        self.paths.sort()
        self.idx = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.idx < len(self.paths):
            img = cv2.imread(self.paths[self.idx])
            # print(img.shape)
            # Resize to fit input
            img = cv2.resize(img, (416,416))
            # CHW ? 
            img = img.transpose((2,0,1))
            img = np.expand_dims(img, axis=0)
            self.idx += 1
            return img
        else:
            raise StopIteration
    def __len__(self):
        return len(self.paths)

def calibrate_dataset(path, num):
    dataset = Dataset(path)
    for i, data in enumerate(dataset):
        # print(i)
        if i > num:
            break
        yield {"images": data}

def model_check():
    base_path = '/home/xinyuwang/adehome/tvm_example/compiled_models/'
    model_path = base_path + 'deploy_lib.tar'
    # lib_path = base_path + 'deploy_lib.so'
    # graph_path = base_path + 'deploy_graph.json'
    # params_path = base_path + 'deploy_param.params'

    dev = tvm.device('llvm', 0)
    lib = tvm.runtime.load_module(model_path)
    module = graph_executor.GraphModule(lib["default"](dev))

    onnx_ = onnx.load(yolox)
    input_name = "images"
    output_shape = (1,3549,85)

    x = np.ones((1,3,416,416), dtype=np.float32)
    # x = np.random.uniform(-1, 1, size=(1,3,416,416)).astype("float32")
    module.set_input(input_name, x)
    module.run()
    out_tvm = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

    ort_sess = ort.InferenceSession(onnx_.SerializeToString())
    out_onnx = ort_sess.run(None, {input_name: x})

    result = out_onnx[0] - out_tvm
    print(result[np.where(result > 0.001)])
    print(len(result[np.where(result > 0.001)]))


def quantize():
    onnx_ = onnx.load(yolox)
    x = np.ones((1,3,416,416), dtype=np.float32)
    input_name = "images"

    path = '/home/xinyuwang/adehome/tvm_example/python/coco-2017/test/data'
    output_path = '/home/xinyuwang/adehome/tvm_example/compiled_models/'

    target = tvm.target.create('llvm')

    shape_dict = {input_name: x.shape}

    mod, params = relay.frontend.from_onnx(onnx_, shape_dict)
    with relay.quantize.qconfig(calibrate_mode="kl_divergence", weight_scale="max"):
        mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset(path, 100))
        with tvm.transform.PassContext(opt_level=3, config={}):
            mod = relay.build(mod, target=target, params=params)
            mod.export_library(output_path + 'deploy_lib.tar')
            # graph = mod.get_graph_json()
            # params = mod.get_params()

            # with open(output_path + 'deploy_graph.json', 'w', encoding='utf-8') as graph_file:
            #     graph_file.write(graph)

            # with open(output_path + 'deploy_param.params', 'wb') as param_file:
            #     param_file.write(relay.save_param_dict(params))

def no_quantize():
    onnx_ = onnx.load(yolox)
    x = np.ones((1,3,416,416), dtype=np.float32)
    input_name = "images"

    path = '/home/xinyuwang/adehome/tvm_example/python/coco-2017/test/data'
    output_path = '/home/xinyuwang/adehome/tvm_example/compiled_models/'

    target = tvm.target.create('llvm')

    shape_dict = {input_name: x.shape}

    mod, params = relay.frontend.from_onnx(onnx_, shape_dict)
    with tvm.transform.PassContext(opt_level=3, config={}):
        mod = relay.build(mod, target=target, params=params)
        mod.export_library(output_path + 'deploy_lib_original.tar')

if __name__ == "__main__":
    # no_quantize()
    quantize()
    model_check()
