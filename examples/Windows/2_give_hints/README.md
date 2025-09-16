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

// If the shim is implemented and enabled by replacing "nvcuda.dll"
// with which compiled with XSched,
// XSched will automatically intercept the driver API calls and 
// convert them to XQueue API calls.
// Here, kernel launch on this stream (HwQueue) will be intercepted
// by nvcuda.dll and submit to the created XQueue.
//                    cuda runtime                   nvcuda.dll
// kernel <<<...>>>() ------------> cuLaunchKernel() ----------> XQueueSubmit()
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

```psl
# go to the root directory of XSched
cd xsched
# by default, XSched will be installed to xsched/output
make cuda INSTALL_PATH=<install_path>
```

For this example, just use the Makefile to build the app.

```psl
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
target_link_libraries(<your_target> XSched::nvcuda)
```

- Link manually

```psl
# build XSched first and link XSched libraries
nvcc -o app_with_hints app_with_hints.cu -I<install_path>/include -L<install_path>/lib -lnvcuda
```

## Run the App with XSched

Step 0: Refer to the [API Interception](../README.md) section to set up the API interception.

Step 1: Start the XSched server (xserver)

```psl
<install_path>\bin\xserver.exe HPF 50000
```

Step 2: Set environment variables

```psl
# use the global (GLB) scheduler, i.e., the xserver
$env:XSCHED_SCHEDULER = "GLB"
```

Step 3: Run the app

```psl
# in the first terminal
# 1: the priority of the XQueue, bigger is higher priority
.\app_with_hints.exe 1

# in the second terminal, set the priority to 0
.\app_with_hints.exe 0
```

You will see the output like this:

```psl
# In the first terminal
Task 38 completed in 75 ms
Task 39 completed in 84 ms
Task 40 completed in 75 ms
Task 41 completed in 81 ms
Task 42 completed in 75 ms
Task 43 completed in 82 ms
Task 44 completed in 75 ms
Task 45 completed in 85 ms
Task 46 completed in 76 ms
```

```psl
# In the second terminal
Task 0 completed in 235 ms
Task 1 completed in 216 ms
Task 2 completed in 202 ms
Task 3 completed in 237 ms
Task 4 completed in 215 ms
Task 5 completed in 215 ms
Task 6 completed in 215 ms
```

The results are similar to example 1_transparent_sched.
You can also use `xcli` to give hints to the xserver to adjust the priority of the XQueue.
See [1_transparent_sched](../1_transparent_sched/README.md#xcli) for more details.
