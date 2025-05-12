# XPreempt

The XPreempt library mainly includes following parts:

- `HwCommand & HwQueue`: Base classes abstraction for hardware command and queue. They are specifically implemented in `platforms` for different hardware.
- `XQueue`: The main abstraction for task controlling and scheduling.
- `SchedAgent`: An agent that watches the status of XQueues and handles scheduling events.

## HwCommand & HwQueue

`HwCommand` is an encapsulation of driver API calling. We use this kind of abstraction to buffer API calling in XQueue, and use scheduler to decide when to launch it to `HwQueue`.

`HwQueue` is an encapsulation of hardware queue, e.g., `cudaStream` in CUDA, `hipStream` in ROCm, `ze_command_queue` in LevelZero, etc. When a `HwCommand` is launched to `HwQueue`, it will be executed in the hardware queue by calling respective driver API.

## XQueue

`XQueue` is a queue of XPU commands that are executed sequentially in the order of submission. `XQueue` is one-to-one mapped to a `HwQueue`.

## SchedAgent

When process starts, it will load `XPreempt` lib and initilize a `SchedAgent` with corresponding `Executor` and `Scheduler`.

- `Executor` implements `Execute` function to handle `Operation` that `SchedAgent` receives.
- `Scheduler` is an instance of `sched::Scheduler`. It will bind its `executor_` with `Executor`.

When `SchedAgent` receives `Event` from `XCtrl` or `XQueue`, it will call `RecvEvent` of `Scheduler` to handle.

By default, the type of `Scheduler` is `GlobalScheduler`(details in `sched::Scheduler`), it means `Scheduler` will notify the daemon scheduler via IPC whem `SchedAgent` receives events rather than handle them directly. After daemon scheduler finish handling events, it will generate operations and send back to `Scheduler`, then `Scheduler` call `executor_`(bound with `Executor`) to execute these operations, which includes resume/suspend `XQueue` or change configuation of `XQueue`.
