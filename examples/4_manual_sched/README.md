# Schedule Manually using Suspend and Resume APIs

## Get Started

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

// If the shim is implemented and enabled by setting
// the "LD_PRELOAD" or "LD_LIBRARY_PATH" environment variable,
// XSched will automatically intercept the driver API calls and 
// convert them to XQueue API calls.
// Here, kernel launch on this stream (HwQueue) will be intercepted
// by libshimcuda.so and submit to the created XQueue.
//                     cuda runtime                  libshimcuda.so
// kernel <<<...>>>() ------------> cuLaunchKernel() -------------> XQueueSubmit()
kernel<<<grid, block, 0, stream>>>(...);

// You can manually suspend or resume the XQueue (application managed scheduling mode)
// In this mode, the scheduler will not make any scheduling decision.
// To enable this mode, you need to set the "XSCHED_SCHEDULER" environment variable to "APP",
// e.g., export XSCHED_SCHEDULER=APP.
// See "protocol/include/xsched/protocol/def.h" for more details.
XQueueSuspend(xq);
XQueueResume(xq);

// destroy the XQueue
XQueueDestroy(xq);

// destroy the HwQueue
HwQueueDestroy(hwq);
```

## Link XSched

- Use CMake

```cmake
# option 1: build XSched first and use path hints to find XSched
# or use cmake -DCMAKE_PREFIX_PATH=<install_path>/lib/cmake instead
find_package(XSched REQUIRED HINTS "<install_path>/lib/cmake")

# option 2: use absolute path to add XSched as subdirectory
add_subdirectory(<xsched_path> xsched)

... # add your target

# link XSched libraries
target_link_libraries(<your_target> XSched::preempt XSched::halcuda ...)
```

- Link manually

```bash
# build XSched first and link XSched libraries
nvcc -o app app.cu -I<install_path>/include -L<install_path>/lib -lpreempt -lhalcuda ...
```

## Run apps with XSched
