# Protocol

`protocol` defines macros and constants used in XSched.
It also provides helper functions to read these values.

The following configurations can be changed with environment variables.

## XQueue configuration

| Name                           | Type   | Range          | Default Value | Description                                                                                                        |
| ------------------------------ | ------ | -------------- | ------------- | ------------------------------------------------------------------------------------------------------------------ |
| XSCHED_AUTO_XQUEUE             | string | ON/OFF         | OFF           | If it automatically creates XQueue when creating a hardware queue.                                                 |
| XSCHED_AUTO_XQUEUE_LEVEL       | int    | [1, 3]         | 1             | XQueue preemption level.                                                                                           |
| XSCHED_AUTO_XQUEUE_THRESHOLD   | int    | [1, MAX_INT64] | 16            | Maximum number of in-flight commands that XSched can launch to the native queue. |
| XSCHED_AUTO_XQUEUE_BATCH_SIZE  | int    | [1, threshold] | 8             | Number of queued commands launched in one batch. |
| XSCHED_AUTO_XQUEUE_PRIORITY    | int    | [-256, 255]    | 0             | Default priority for the `HPF` policy. Bigger values mean higher priority. |
| XSCHED_AUTO_XQUEUE_UTILIZATION | int    | [0, 100]       | 100           | Default utilization share for the `UP` and `PUP` policies. |
| XSCHED_AUTO_XQUEUE_TIMESLICE   | int    | [100, 100000]  | 5000          | Default time slice for round-robin-style policies, in microseconds. |
| XSCHED_AUTO_XQUEUE_DEADLINE    | int    | [1, MAX_INT64] | unset         | Default deadline hint for deadline-based policies. |
| XSCHED_AUTO_XQUEUE_KDEADLINE   | int    | [1, MAX_INT64] | unset         | Default k-deadline hint for `KEDF`. |
| XSCHED_AUTO_XQUEUE_LAXITY      | int    | [1, MAX_INT64] | unset         | Default laxity hint for `LAX`. |

## Platform Library Overrides

Platform adapters normally find their native runtime libraries from the system
search path and common installation paths. These variables can override the
library path when needed:

| Name |
| ---- |
| XSCHED_ASCEND_LIB |
| XSCHED_COREX_LIB |
| XSCHED_COREX_RT_LIB |
| XSCHED_CUDA_LIB |
| XSCHED_CUPTI_LIB |
| XSCHED_CUDART_LIB |
| XSCHED_CUDLA_LIB |
| XSCHED_HIP_LIB |
| XSCHED_LEVELZERO_LIB |
| XSCHED_OPENCL_LIB |
| XSCHED_VPI_LIB |

## CUDA XQueue-Specific Configuration

| Name                           | Type   | Range          | Default Value | Description                                                                                                        |
| ------------------------------ | ------ | -------------- | ------------- | ------------------------------------------------------------------------------------------------------------------ |
| XSCHED_CUDA_SINGLE_STREAM_PER_PROCESS | string | ON/OFF | OFF | Use one internal CUDA stream per process. |
| XSCHED_CUDA_LV3_IMPL           | string | TSG/TRAP       | TRAP          | Preemption method of Level-3 XQueue for CUDA. |

## Level Zero XQueue-Specific Configuration

| Name                           | Type   | Default Value | Description |
| ------------------------------ | ------ | ------------- | ----------- |
| XSCHED_LEVELZERO_SLICE_CNT     | int    | platform default | Command slicing count used by the Level Zero adapter. |

## Scheduler configuration

Change the scheduler type by setting `XSCHED_SCHEDULER`.
The default value is `APP`.

| Value | Full Name           | Transparency | Description                                                                              |
| ----- | ------------------- | ------------ | ---------------------------------------------------------------------------------------- |
| LCL   | Local Scheduler     | ✅            | Process has its own scheduler, it only schedules xqueues created by itself.              |
| GLB   | Global Scheduler    | ✅            | The process uses the global scheduler in XServer.                                        |
| APP   | Application Managed | ❌            | The application uses XQueue APIs to manage XQueues directly.                             |

Scheduler suspend behavior can be adjusted with:

| Name | Type | Default Value | Description |
| ---- | ---- | ------------- | ----------- |
| XSCHED_SCHEDULER_SUSPEND_SYNC_HWQ | string | OFF | Synchronize the native hardware queue during suspend. |

## Policy configuration

Set `XSCHED_POLICY` to create a local scheduler with the selected policy.
If it is set, `XSCHED_SCHEDULER` is ignored in that process.
XServer also accepts a policy name as its command-line argument.

| Value | Full Name |
| ----- | --------- |
| HPF   | Highest Priority First |
| HHPF  | Heterogeneous Highest Priority First |
| CHPF  | CPU Highest Priority First |
| UP    | Utilization Partition |
| PUP   | Process Utilization Partition |
| SPUP  | Strict Process Utilization Partition |
| KEDF  | K-Earliest Deadline First |
| LAX   | Laxity-based |
| AWF   | Active Window First |
| CFS   | Completely Fair Scheduler |
| MLFQ  | Multi-Level Feedback Queue |

*(refer to [policies](../sched/README.md))*
