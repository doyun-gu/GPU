import numpy as np
import cupy as cp
from time import time

def benchmark_processor(arr, func, argument):
    start_time = time()
    func(arr, argument)  # your argument will be broadcasted into a matrix automatically
    finish_time = time()
    elapsed_time = finish_time - start_time
    return elapsed_time

# load a matrix to global memory
array_cpu = np.random.randint(0, 255, size=(9999, 9999))

# load the same matrix to GPU memory
array_gpu = cp.asarray(array_cpu)

# benchmark matrix addition on CPU by using a NumPy addition function
cpu_time = benchmark_processor(array_cpu, np.add, 999)

# Run a pilot iteration on GPU first to compile and cache the function kernel on GPU
_ = benchmark_processor(array_gpu, cp.add, 1)  # This is just to warm up and compile the kernel

# benchmark matrix addition on GPU by using a CuPy addition function
gpu_time = benchmark_processor(array_gpu, cp.add, 999)

# Determine how much faster the GPU is compared to the CPU
if cpu_time > gpu_time:
    faster_processor = "GPU"
    speed_increase = (cpu_time - gpu_time) / gpu_time * 100
else:
    faster_processor = "CPU"
    speed_increase = (gpu_time - cpu_time) / cpu_time * 100

print(f"CPU time: {cpu_time} seconds\nGPU time: {gpu_time} seconds.\n{faster_processor} was {speed_increase}% faster.")
