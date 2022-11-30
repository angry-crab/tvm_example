import tvm
import numpy as np
from tvm.ir.module import IRModule
from tvm.script import tir as T
from tvm import te

pillar_features = te.placeholder((40000, 32), name="pillar_features", dtype="float32")
coords = te.placeholder((40000, 3), name="coords", dtype="int32")
# spatial_features = te.placeholder((32*560*560), name="spatial_features", dtype="float32")
# i = te.thread_axis((0, 40000), name="i")
# j = te.thread_axis((0, 32), name="j")
f = lambda i,j,k: te.if_then_else()
func = te.compute((32,560,560), f, name="spatial_features")