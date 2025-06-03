# XPreempt

## xqueue

The XPreempt library implements the XQueue abstraction, the main functions are listed in following table, including submit and wait to support task execution, and suspend and resume to schedule XQueues. Commands submitted to the XQueue are buffered and launched to the XPU at a proper time to achieve task preemption.

| XQueue Interface | Description                                   |
| ---------------- | --------------------------------------------- |
| submit(xq, cmd)  | Submit a command (cmd) to XQueue (xq)         |
| wait(xq, cmd)    | Wait for a given cmd in xq to complete        |
| wait_all(xq)     | Wait for all cmds in xq to complete           |
| suspend(xq)      | Suspend xq to pause task execution resume(xq) |
| Resume(xq)       | to continue task execution                    |

## hal

`hal` provide `HwCommand` and `HwQueue` abstraction, it only declared the meanings of each interface, specific implementation are finished in `platforms`. Besides, `hal` also maintained global variables `HwQueueManager` and `HwCommandManager`, they manage all `HwCommand` and `HwQueue` in the process.

## sched

The XPreempt library also contains an agent that watches the status of XQueues (e.g., ready or idle). When process starts, it will load `XPreempt` lib and initilize a `SchedAgent` with corresponding `Executor` and `Scheduler`.

- `Executor` implements `Execute` function to handle `Operation` that `SchedAgent` receives.
- `Scheduler` is an instance of `sched::Scheduler`. It will bind its `executor_` with `Executor`.

When `SchedAgent` receives `Hint` from `XCLI` or `Event` from `XQueue`, it will call `RecvEvent` of `Scheduler` to handle.

By default, the type of `Scheduler` is `GlobalScheduler`(details in `sched::Scheduler`), it means `Scheduler` will notify the daemon scheduler via IPC whem `SchedAgent` generates scheduling events rather than handle them directly. After daemon scheduler finish handling events, it will generate operations and send back to `Scheduler`, then `Scheduler` call `executor_`(bound with `Executor`) to execute these operations.
