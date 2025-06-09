# Schedule by Manually Giving Hints to XSched

## Usage

```c
#include "xsched/xsched.h"
#include "xsched/cuda/hal.h" // use the hal header of the target platform

// create a cuda stream
cudaStream_t stream;
cudaStreamCreate(&stream);

// warp this cuda stream with a HwQueue (CudaQueue)
HwQueueHandle hwq;
CudaQueueCreate(&hwq, stream);

// create an XQueue using this HwQueue
// kPreemptLevelBlock     : level-1
// kPreemptLevelDeactivate: level-2
// kPreemptLevelInterrupt : level-3
XQueueHandle xq;
XQueueCreate(&xq, hwq, kPreemptLevelBlock, kQueueCreateFlagNone);

// Set the threshold and command batch size of the XQueue.
// See "XSCHED_AUTO_XQUEUE_THRESHOLD" and "XSCHED_AUTO_XQUEUE_BATCH_SIZE"
// in 1_transparent_sched/README.md for more details.
XQueueSetLaunchConfig(xq, 8, 4);

// If the shim is implemented and enabled by setting
// the "LD_PRELOAD" or "LD_LIBRARY_PATH" environment variable,
// XSched will automatically intercept the driver API calls and 
// convert them to XQueue API calls.
// Here, kernel launch on this stream (HwQueue) will be intercepted
// by libshimcuda.so and submit to the created XQueue.
//                     cuda runtime                  libshimcuda.so
// kernel <<<...>>>() ------------> cuLaunchKernel() -------------> XQueueSubmit()
kernel<<<grid, block, 0, stream>>>(...);

// Give hints to set priority for this XQueue using hint API.
// The hints will be sent to the policy to make scheduling decisions.
XHintPriority(xq, priority);

// destroy the XQueue
XQueueDestroy(xq);

// destroy the HwQueue
HwQueueDestroy(hwq);
```

## Build the App with XSched

Make sure you have already built XSched with CUDA support.

```bash
# go to the root directory of XSched
cd xsched
# by default, XSched will be installed to xsched/output
make cuda INSTALL_PATH=<install_path>
```

For this example, just use the Makefile to build the app.

```bash
make
```

### Link XSched with Your Own App

- Use CMake

```cmake
# option 1: build XSched first and use path hints to find XSched
# or use cmake -DCMAKE_PREFIX_PATH=<install_path>/lib/cmake instead
find_package(XSched REQUIRED HINTS "<install_path>/lib/cmake")

# option 2: use absolute path to add XSched as subdirectory
add_subdirectory(<xsched_path> xsched)

... # add your target

# link XSched libraries
target_link_libraries(<your_target> XSched::preempt XSched::halcuda)
```

- Link manually

```bash
# build XSched first and link XSched libraries
nvcc -o app_with_hints app_with_hints.cu -I<install_path>/include -L<install_path>/lib -lpreempt -lhalcuda
```

## Run the App with XSched

Step 1: Start the XSched server (xserver)

```bash
<install_path>/bin/xserver HPF 50000
```

Step 2: Set environment variables

```bash
# use the global (GLB) scheduler, i.e., the xserver
export XSCHED_SCHEDULER=GLB

# Intercept the CUDA calls using the shim library
export LD_LIBRARY_PATH=<install_path>/lib:$LD_LIBRARY_PATH
```

Step 3: Run the app

```bash
# in the first terminal
# 1: the priority of the XQueue, bigger is higher priority
./app_with_hints 1

# in the second terminal, set the priority to 0
./app_with_hints 0
```

You will see the output like this:

```
# In the first terminal
Task 80 completed in 67 ms
Task 81 completed in 71 ms
Task 82 completed in 67 ms
Task 83 completed in 73 ms
Task 84 completed in 67 ms
Task 85 completed in 73 ms
```

```
# In the second terminal
Task 20 completed in 182 ms
Task 21 completed in 161 ms
Task 22 completed in 195 ms
Task 23 completed in 193 ms
Task 24 completed in 173 ms
Task 25 completed in 185 ms
```

The results are similar to example 1_transparent_sched.
You can also use `xcli` to give hints to the xserver to adjust the priority of the XQueue.
See [1_transparent_sched](../1_transparent_sched/README.md#xcli) for more details.
