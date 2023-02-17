import pycuda.autoinit
import pycuda.driver as cuda

dev=cuda.Device(0)
# print(dev.get_attributes())
print(dev.max_shared_memory_per_block)
print(dev.max_threads_per_block)
print(dev.warp_size)