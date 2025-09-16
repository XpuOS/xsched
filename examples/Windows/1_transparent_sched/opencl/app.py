import numpy as np
import pyopencl as cl
import time

W = 800  # Matrix size
N = 10000  # Number of tasks

# Print available platforms
platforms = cl.get_platforms()
for platform_idx, platform in enumerate(platforms):
    print(f"Platform [{platform_idx}]:", platform.name)
    devices = platform.get_devices()
    for device_idx, device in enumerate(devices):
        print(f"  Device [{platform_idx}.{device_idx}]:", device.name)

# Select the first platform and device
platform_idx = 1
device_idx = 0
platform = cl.get_platforms()[platform_idx]
device = platform.get_devices()[device_idx]
print("Using platform [%d]: %s" % (platform_idx, platform.name))
print("Using device [%d.%d]: %s" % (platform_idx, device_idx, device.name))

# Create context and queue
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Create input matrix
A = np.random.rand(W, W).astype(np.float32)
B = np.random.rand(W, W).astype(np.float32)
C = np.zeros((W, W), dtype=np.float32)

# Create OpenCL buffer
mf = cl.mem_flags
A_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
B_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
C_buf = cl.Buffer(context, mf.WRITE_ONLY, C.nbytes)

# Compile the Kernel
kernel_code = """
__kernel void matrix_mul(__global float* A, __global float* B, __global float* C, int width) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float sum = 0.0;
    for (int k = 0; k < width; k++) {
        sum += A[i * width + k] * B[k * width + j];
    }
    C[i * width + j] = sum;
}
"""
program = cl.Program(context, kernel_code).build()

# Set kernel parameters
kernel = cl.Kernel(program, "matrix_mul")
kernel.set_args(A_buf, B_buf, C_buf, np.int32(W))
global_size = (W, W)
local_size = (1, 1)

# Execute the kernel multiple times and measure execution time
for i in range(N):
    start_time = time.time()
    cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
    cl.enqueue_copy(queue, C, C_buf)
    queue.finish()
    end_time = time.time()

    print("Task %d completed in %.6f seconds" % (i, end_time - start_time))
