# Sched

`Sched` implements code of `XSched Daemon`.

## policy

`policy` implements scheduling policy. `policy.cpp` provide a uniform abstract `sched::policy`, `edf.cpp`, `hpf.cpp`, `lax.cpp` and `up.cpp` are specific strategies.

To realize a policy, you must need to finish `Sched` and `RecvHint`. `Sched` receive a parameter `Status`, which includes all xqueues information. You can use `Suspend` and `Resume` API defined in `policy.cpp` to suspend or resume xqueues to achieve scheduling effect. `RecvHint` receive a parameter `Hint`, it is used to change parameters in policies.

## protocol

The `protocol` part in `sched` defines some data structures.

- `event` are used in communication between `preempt::SchedAgent` and `sched::Scheduler`. Each agent monitors the status change events of the XQueues in its process, e.g. become ready when new commands are submitted and idle when all commands are completed. These events are sent to the scheduler daemon to maintain a global mirror of XQueue status (ready or idle, XPU device ID, process ID, etc.) and trigger the policy upon status changes.
- `hint` are customized data structures to change policy parameters.
- `names` contains some macro definitions about scheduling policy name.
- `operation` is used by `sched::Scheduler` to send to `preempt::SchedAgent` to enforces the decisions of the policy.
- `status` is a data structure saving xqueue information.

## scheduler

When `preempt::SchedAgent` initilize, it will create a scheduler for its process. There are three types of scheduler, `LocalScheduler`, `GlobalScheduler` and `AppManagedScheduler`. User can change type by specifying environment variables `XSCHED_POLICY`(The default is `GlobalScheduler`).

### LocalScheduler

`local.cpp` implements processing logic of a scheduler. When running a `LocalScheduler`, it will start a thread running function `Worker`. `Worker` contains a loop to take events in `event_queue_` and call `Sched` API of policy. `local.cpp` also defines following function:

- `Suspend` and `Resume` are called by `policy` to really realize control to xqueues
- `SetPolicy` changes policy used in `LocalScheduler`

If a process's scheduler type is `LocalScheduler`, it means the process can only schedule xqueues created by itself.

### GlobalScheduler

Actually, `global.cpp` only achieves two IPC channel. Its `Worker` thread just receive `Operation` by `rece_chan_` and call `Execute` bound when created(refering to `AddExecutor` of `preempt::SchedExecutor` in file `preempt/src/sched/executor.cpp`).

- `send_chan_` uses name `XSCHED_SERVER_CHANNEL_NAME` and its type is sender(receiver is `recv_chan_` running in `service/server`). When `sched::SchedAgent` call `RecvEvent` to force scheduler to receive events, `send_chan_` will send event to receiver and do nothing else.
- `rece_chan_` uses name bound to its PID and its type is receiver(senders is `client_chan` in `service/server`).

If a process wants to use `GlobalScheduler`, user must first run a global server(refering to `service::Server` in floder `service/server`). Server will create a `LocalScheduler` and two IPC channels to communicate with process's scheduler, receiving events sended by process's `send_chan_` and generating operations by policy in its `LocalScheduler`, then send these operations back to process's `rece_chan_`.

### AppManagedScheduler

`AppManagedScheduler` is a empty scheduler. User need to manually add and execute `Suspend` and `Resume` API in `xctrl`(refering to `platfform/shim/xctrl.h`) to control xqueues's behavior.
