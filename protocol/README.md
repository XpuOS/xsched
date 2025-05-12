# Protocol

`protocol` defines macro and constant parameters used in XSched, also provide some functions to get these constant parameters.

It also defines configurations that can be changed by command "export Name=VALUE" in shell.

## XQueue configuration

| Name                           | Type   | Range          | Default Value | Description                                                                                                        |
| ------------------------------ | ------ | -------------- | ------------- | ------------------------------------------------------------------------------------------------------------------ |
| XSCHED_AUTO_XQUEUE             | string | ON/OFF         | OFF           | If it automatically creates XQueue when creating a hardware queue.                                                 |
| XSCHED_AUTO_XQUEUE_LEVEL       | int    | [1, 3]         | 1             | XQueue preemption level.                                                                                           |
| XSCHED_AUTO_XQUEUE_THRESHOLD   | int    | [1, MAX_INT64] | 16            | Maximum number of commands for In-flight status(commands that can be executed simultaneously in a hardware queue). |
| XSCHED_AUTO_XQUEUE_BATCH_SIZE  | int    | [1, threshold] | 8             | TODO                                                                                                               |
| XSCHED_AUTO_XQUEUE_PRIORITY    | int    | [-256, 255]    | 0             | Default  priority  of   "HPF" policy                                                                               |
| XSCHED_AUTO_XQUEUE_UTILIZATION | int    | [0, 100]       | 100           | Default  utilization  of   "UP" and "PUP" policy                                                                   |
| XSCHED_AUTO_XQUEUE_TIMESLICE   | int    | [100, 100000]  | 5000          | Default  time slice  of   "RR" policy                                                                              |

## Policy configuration

Change default scheduling policy in XServer by setting `XSCHED_POLICY` to policy name. The default value is `HPF`.

| Value | Description                   |
| ----- | ----------------------------- |
| HPF   | Highest Priority First        |
| AMG   | Application Managed           |
| RR    | Round Robin                   |
| UP    | Utilization Partition         |
| PUP   | Process Utilization Partition |
| EDF   | Earliest Deadline First       |
| LAX   | Laxity-based                  |

**NOTE:**
**There is a special value of "GLB", which means "Global Scheduler". It is not a specific scheduling policy and cannot be used for XServer. It is used for applications.**
**Before starting an application, user must set `XSCHED_POLICY` to "GLB" to use the global scheduler. Otherwise, application will create a local scheduler which can only schedule xqueues created by the application itself.**
**We offer this option only to provide convenience for developers during single process testing.**
