# Protocol

`protocol` defines macro and constant parameters used in XSched, also provide some functions to get these constant parameters.

It also defines configurations that can be changed by command `export Name=VALUE` in shell.

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

## Scheduler configuration

Change default scheduler type of processes by setting `XSCHED_SCHEDULER` to type name. The default value is `GLB`.

| Value | Full Name           | Transparency | Description                                                                              |
| ----- | ------------------- | ------------ | ---------------------------------------------------------------------------------------- |
| LCL   | Local Scheduler     | ✅            | Process has its own scheduler, it only schedules xqueues created by itself.              |
| GLB   | Global Scheduler    | ✅            | Process use global scheduler in XServer.                                                 |
| AMG   | Application Managed | ❌            | There is no scheduler in process, application needs to use XQueue API to manage xqueues. |

## Policy configuration

Change default scheduling policy in XServer by setting `XSCHED_POLICY` to policy name. The default value is `HPF`.

| Value | Full Name                   |
| ----- | ----------------------------- |
| HPF   | Highest Priority First        |
| UP    | Utilization Partition         |
| PUP   | Process Utilization Partition |
| EDF   | Earliest Deadline First       |
| LAX   | Laxity-based                  |

*(refering to [policies](../sched/README.md))*
