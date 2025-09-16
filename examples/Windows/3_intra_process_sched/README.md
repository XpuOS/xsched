# Schedule XQueues within one Process

## Usage

```c
#include "xsched/xsched.h"
#include "xsched/cuda/hal.h" // use the hal header of the target platform

// create two cuda streams and two XQueues
cudaStream_t stream_1, stream_2;
cudaStreamCreate(&stream_1);
cudaStreamCreate(&stream_2);

HwQueueHandle hwq_1, hwq_2;
CudaQueueCreate(&hwq_1, stream_1);
CudaQueueCreate(&hwq_2, stream_2);

XQueueHandle xq_1, xq_2;
XQueueCreate(&xq_1, hwq_1, kPreemptLevelBlock, kQueueCreateFlagNone);
XQueueCreate(&xq_2, hwq_2, kPreemptLevelBlock, kQueueCreateFlagNone);

// use hints to set the priority of each XQueue
XHintPriority(xq_1, 1); // lower priority
XHintPriority(xq_2, 2); // higher priority

// Set the scheduler to local scheduler so that the process will use
// the in-process scheduler, rather than the xserver.
// Set the policy of the local scheduler to highest priority first.
XHintSetScheduler(kSchedulerLocal, kPolicyHighestPriorityFirst);

// Use the stream to launch commands (kernels & memory copies) as usual.
// The task on stream_2 can preempt the task on stream_1 as it has higher priority.
// e.g., on thread 1:
kernel<<<grid, block, 0, stream_1>>>(...);
// e.g., on thread 2:
kernel<<<grid, block, 0, stream_2>>>(...);
```

## Build the App with XSched

```bash
# go to the root directory of XSched and build XSched with CUDA support
cd xsched
# by default, XSched will be installed to xsched/output
make cuda INSTALL_PATH=<install_path>

# build the app
cd examples/Windows/3_intra_process_sched
make
```

Please refer to [2_give_hints](../2_give_hints/README.md#link-xsched-with-your-own-app) for guidance to link XSched with your own app.

## Run the App with XSched

Refer to the [API Interception](../README.md) section to set up the API interception, then run the app.

```bash
./app_concurrent
```

You will see the output like this:

```psl
[INFO @ T22828 @ 15:16:11.706551] using app-managed scheduler
[INFO @ T22828 @ 15:16:11.709107] using local scheduler with policy HPF
[INFO @ T7976 @ 15:16:12.117797] XQueue (0x4c48011d5b0fde70) from process 19528 created
[INFO @ T7976 @ 15:16:12.118742] set priority 2 for XQueue 0x4c48011d5b0fde70
Prio 2 Task 0 completed in 78 ms
Prio 2 Task 1 completed in 75 ms
Prio 2 Task 2 completed in 76 ms
[INFO @ T7976 @ 15:16:12.988397] XQueue (0x4c48011d5b0fe730) from process 19528 created
[INFO @ T7976 @ 15:16:12.989549] set priority 1 for XQueue 0x4c48011d5b0fe730
Prio 2 Task 7 completed in 101 ms
Prio 2 Task 8 completed in 78 ms
Prio 1 Task 0 completed in 226 ms
Prio 2 Task 9 completed in 75 ms
Prio 2 Task 10 completed in 78 ms
Prio 1 Task 1 completed in 182 ms
Prio 2 Task 11 completed in 76 ms
Prio 2 Task 12 completed in 80 ms
```

The tasks are identical, but the higher-priority tasks have significant lower latencies compared to the lower-priority ones.
