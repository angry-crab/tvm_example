import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm

proto = "/home/xinyuwang/adehome/baidu_cnn/vls-128.prototxt"
caffemodel = "/home/xinyuwang/adehome/baidu_cnn/vls-128.caffemodel"

