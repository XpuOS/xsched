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

// use the stream to launch commands (kernels & memory copies) as usual
kernel<<<grid, block, 0, stream>>>(...);

// You can manually suspend or resume the XQueue (application managed scheduling mode).
// In this mode, the scheduler will not make any scheduling decisions and apps
// are responsible for scheduling the XQueue.
// If "XSCHED_SCHEDULER" and "XSCHED_POLICY" environment variables are not set,
// XSched will by default use this application-managed scheduling mode.
// You can also set "XSCHED_SCHEDULER" to "APP", or use XHintSetScheduler()
// to enable this mode.
// e.g., export XSCHED_SCHEDULER=APP, or
XHintSetScheduler(kSchedulerAppManaged, kPolicyUnknown);

// manually suspend or resume the XQueue
XQueueSuspend(xq);
XQueueResume(xq);
```

The usage on HIP platform is similar.

## Build the App with XSched

```bash
# go to the root directory of XSched and build XSched with CUDA support
cd xsched
# by default, XSched will be installed to xsched/output
# on CUDA platform
make cuda INSTALL_PATH=<install_path>
# on HIP platform
make hip INSTALL_PATH=<install_path>

# build the app
cd examples/4_manual_sched
# on CUDA platform
make cuda
# on HIP platform
make hip
```

Please refer to [2_give_hints](../2_give_hints/README.md#link-xsched-with-your-own-app) for guidance to link XSched with your own app.

## Run the App with XSched

```bash
# Intercept the CUDA calls using the shim library
export LD_LIBRARY_PATH=<install_path>/lib:$LD_LIBRARY_PATH
# Intercept the HIP calls using the shim library
export LD_PRELOAD=<install_path>/lib/libshimhip.so:$LD_PRELOAD

./app_sched
```

You will see the output like this:

```
high prio Task 10 completed in 66 ms
high prio Task 11 completed in 69 ms
low  prio Task 14 completed in 172 ms
high prio Task 12 completed in 66 ms
high prio Task 13 completed in 70 ms
low  prio Task 15 completed in 159 ms
high prio Task 14 completed in 66 ms
high prio Task 15 completed in 69 ms
low  prio Task 16 completed in 184 ms
```

The application itself suspends the low-priority task when the high-priority task starts,
and resumes the low-priority task when the high-priority task finishes.
Thus, the higher-priority tasks have significant lower latencies compared to the lower-priority ones.
