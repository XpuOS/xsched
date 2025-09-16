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

// place launch commands (kernels & memory copies) into callback
XResult callback(HwQueueHandle queue, void *params) {
    ...
    kernel<<<grid, block, 0, stream>>>(...);
    ...
}

// create an HwCommand
HwCommandHandle hw_command;
HwCommandCreateCallback(&hw_command, callback, &params);

// use XQueueSubmit to submit the HwCommand to the XQueue
XQueueSubmit(xq, hw_command);

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

## Build the App with XSched

```bash
# go to the root directory of XSched and build XSched with CUDA support
cd xsched
# by default, XSched will be installed to xsched/output
make cuda INSTALL_PATH=<install_path>

# build the app
cd examples/8_nontransparent_sched
make
```

Please refer to [2_give_hints](../2_give_hints/README.md#link-xsched-with-your-own-app) for guidance to link XSched with your own app.

## Run the App with XSched

Refer to the [API Interception](../../README.md) section to set up the API interception, then run the app.

```bash
./app_nontrans_sched
```

You will see the output like this:

```psl
low  prio Task 0 completed in 73 ms
low  prio Task 1 completed in 74 ms
low  prio Task 2 completed in 74 ms
low  prio Task 3 completed in 74 ms
low  prio Task 4 completed in 74 ms
low  prio Task 5 completed in 74 ms
low  prio Task 6 completed in 74 ms
low  prio Task 7 completed in 74 ms
high prio Task 0 completed in 76 ms
low  prio Task 8 completed in 177 ms
high prio Task 1 completed in 74 ms
high prio Task 2 completed in 77 ms
low  prio Task 9 completed in 191 ms
high prio Task 3 completed in 74 ms
high prio Task 4 completed in 76 ms
low  prio Task 10 completed in 207 ms
high prio Task 5 completed in 74 ms
high prio Task 6 completed in 76 ms
low  prio Task 11 completed in 207 ms
```

The application itself suspends the low-priority task when the high-priority task starts,
and resumes the low-priority task when the high-priority task finishes.
Thus, the higher-priority tasks have significant lower latencies compared to the lower-priority ones.
