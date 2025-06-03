# Service

## XClient

`XCLI` is a command-line tool, it is provided to users to change the policy and give scheduling hints(e.g., priorities, deadlines) to the policy in `XServer`.

## XServer

`XServer` consists of following parts:

- `LocalScheduler`(referring to `sched::scheduler`) to execute processing logic.
- Two IPC channels to receive events from `SchedAgent` and send operations to it.
- A http server to comminicate with `XCLI`.
