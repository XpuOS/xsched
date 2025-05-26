# Scheduling Module for XSched

## Schedulers

### Application Managed Scheduler

`AppManagedScheduler` is a empty scheduler. User need to manually add and execute xsched API (refering to [xsched API](../include/README.md)) to control xqueues's behavior.

### Local Scheduler

`local.cpp` implements processing logic of a scheduler. When running a `LocalScheduler`, it will start a thread running function `Worker`. `Worker` contains a loop to take events in `event_queue_` and call `Sched` API of policy. `local.cpp` also defines following function:

- `Suspend` and `Resume` are called by `policy` to really realize control to xqueues
- `SetPolicy` is used to changes policy online

**Note:**

- **If a process's scheduler type is `LocalScheduler`, it means the process can only schedule xqueues created by itself.**
- **The scheduler type of `XServer` must be `GlobalScheduler`.**

### Global Scheduler

<img src="/docs/img/global-scheduler.png" alt="Global Scheduler Framework" width="600" />

Actually, global scheduler only achieves two IPC channel. Its `Worker` thread just receive `Operation` by `rece_chan_` and call `Execute` bound when created (refering to `SetExecutor()` of `sched::Scheduler` in file [scheduler.h](../sched/include/xsched/sched/scheduler/scheduler.h)).

- `send_chan_` uses name `XSCHED_SERVER_CHANNEL_NAME` and its type is sender. When `preempt::SchedAgent` call `RecvEvent` to force scheduler to receive events, `send_chan_` will send event to receiver and do nothing else.
- `rece_chan_` uses name bound to its PID and its type is receiver.

If a process wants to use `GlobalScheduler`, user must first run XServer. XServer will create a `LocalScheduler` and two IPC channels to communicate with process's scheduler, receiving events sended by process's `send_chan_` and generating operations by policy in its `LocalScheduler`, then send these operations back to process's `rece_chan_`.

## Protocol

### Event

`event` are used in communication among `preempt::XQueue`, `preempt::SchedAgent`, `sched::Scheduler`, `service::Server`.

When XQueues' status change, e.g. become ready when new commands enqueue, become idle when all commands are completed, XQueue will send event to `SchedAgent`. `SchedAgent` will process these events by itself or sent them to the XServer according to the type of its scheduler.

### Operation

`operation` is used in communication among `service::Server`, `sched::Scheduler`, `preempt::SchedAgent`.

When policy in `sched::Scheduler` (LocalScheduler) generate scheduling operations, it will call `Execute()` bound to `service::Server`. Then `Execute()` finds corresponding application process and send these operations to `preempt::SchedAgent`.

### Hint

`hint` are customized data structures to change policy parameters. Developers can define their own hint types when implementing a new policy. The hint types we support now are as follows:

| Name                 | Used Policy | Description                                 |
| -------------------- | ----------- | ------------------------------------------- |
| kHintTypePriority    | `HPF`       | Change priority of the xqueue               |
| kHintTypeUtilization | `UP`, `PUP` | Change utilization of the xqueue or process |
| kHintTypeTimeslice   | `UP`, `PUP` | Change time slice                           |
| kHintTypeDeadline    | `EDF`       | Change deadline of the xqueue               |
| kHintTypeLaxity      | `LAX`       | Change laxity of the xqueue.                |

## Policies

`policy.cpp` provide a uniform abstract `sched::policy`, all specific strategies are derived from it.

Now we have implemented several policies, including:

| Value | Full Name                     | Description                                     |
| ----- | ----------------------------- | ----------------------------------------------- |
| HPF   | Highest Priority First        | Firstly run the xqueue with highest priority    |
| UP    | Utilization Partition         | Assign time slices with weights to each xqueue  |
| PUP   | Process Utilization Partition | Assign time slices with weights to each process |
| EDF   | Earliest Deadline First       | Firstly run the xqueue with earliest deadline   |
| LAX   | Laxity-based                  | Firstly run the xqueue with highest laxity      |

## How to Implement Your Own Policy

### 1. Create a new policy class

Create a new policy class in `sched/include/xsched/sched/policy` directory, and inherit from `sched::policy`. Implement the following two functions:

- `Sched` is the main function of the policy, it will be called by `sched::Scheduler` to schedule xqueues.
- `RecvHint` is used to receive hints and change policy parameters.

### 2. Register the new policy

1. Add the new policy type to the enum `PolicyType` in `sched/include/xsched/sched/policy/policy.h`.
2. Add the new policy type to the function `CreatePolicy()` in `sched/src/policy/policy.cpp`.
3. Add the new policy type to the definition of `XSCHED_POLICY` in `protocol/include/xsched/protocol/def.h`.
4. Add the new policy type to the map `PolicyTypeNames` in `sched/src/protocol/names.cpp`.
5. Add the new hint type by need in `hint.h` and `hint.cpp`.
