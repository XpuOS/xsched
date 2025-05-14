# XSched Service

## XClient

`XCLI` is a command-line tool, it is provided to users to change the policy and give scheduling hints(e.g., priorities, deadlines) to the policy in `XServer`.

### Usage

You can run it by `./output/bin/xcli` in the root directory of XSched after compiling.

The parameters it accepts are as follows:

| Usage       | Type   | Description                 | Default    |
| ----------- | ------ | --------------------------- | ---------- |
| `-a,--addr` | string | XSched server ipv4 address. | 127.0.0.1. |
| `-p,--port` | int    | XSched server port.         | 50000.     |

The subcommands it supports are as follows:

<table>
  <tr>
    <th align="center">Subcommand</th>
    <th align="center">Description</th>
    <th align="center">Usage</th>
    <th align="center">Type</th>
    <th align="center">Range</th>
  </tr>
  <tr>
    <td align="center">top</a></td>
    <td align="center">Show the top information of the XQueues</td>
    <td align="center" style="white-space:nowrap">-f,--frequency</td>
    <td align="center">float</td>
    <td align="center">[0,30]</td>
  </tr>
  <tr>
    <td align="center">list</a></td>
    <td align="center">List the information of the XQueues.</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center" rowspan="4">config</a></td>
    <td align="center" rowspan="4">Change the configuration of the XQueues.</td>
    <td align="center" style="white-space:nowrap">-x,--xqueue</td>
    <td align="center">uint64_t</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center" style="white-space:nowrap">-l,--level</td>
    <td align="center" style="white-space:nowrap">int32_t</td>
    <td align="center">[1,3]</td>
  </tr>
  <tr>
    <td align="center" style="white-space:nowrap">-t,--threshold</td>
    <td align="center">int64_t</td>
    <td align="center">[1,65536]</td>
  </tr>
  <tr>
    <td align="center" style="white-space:nowrap">-b,--batch-size</td>
    <td align="center">int64_t</td>
    <td align="center">[1,65536]</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">policy</a></td>
    <td align="center" rowspan="2">Query or set the scheduler policy of the XSched server.</td>
    <td align="center">-q,--query</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center" style="white-space:nowrap">-s -n,--set --name</td>
    <td align="center">string</td>
    <td align="center">XSCHED_POLICY</td>
   </tr>
   <tr>
    <td align="center" rowspan="5">hint</a></td>
    <td align="center" rowspan="5">Give a hint to the XSched server.</td>
    <td align="center">-x,--xqueue</td>
    <td align="center">uint64_t</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center" style="white-space:nowrap">--pid</td>
    <td align="center">pid_t</td>
    <td align="center">-</td>
   </tr>
  <tr>
    <td align="center" style="white-space:nowrap">-p,--priority</td>
    <td align="center">int32_t</td>
    <td align="center">[-255,255]</td>
   </tr>
  <tr>
    <td align="center" style="white-space:nowrap">-u,--utilization</td>
    <td align="center">int32_t</td>
    <td align="center">[0,100]</td>
   </tr>
  <tr>
    <td align="center" style="white-space:nowrap">-t,--timeslice</td>
    <td align="center">int64_t</td>
    <td align="center">[100,100000]</td>
   </tr>
</table>

### Examples

```bash
# Show the top information of the XQueues
./output/bin/xcli top -f 1
# List the information of the XQueues
./output/bin/xcli list
# Change the configuration of the XQueues
./output/bin/xcli config -x xqueue_handle -l 1 -t 16 -b 8
# Query the scheduler policy of the XSched server
./output/bin/xcli policy -q
# Set the scheduler policy of the XSched server
./output/bin/xcli policy -s -n HPF
# Give a hint to the XSched server
./output/bin/xcli hint -x xqueue_handle -p 10
./output/bin/xcli hint --pid pid -u 50
./output/bin/xcli hint -t 2000
```

## XServer

`XServer` consists of following parts:

- `LocalScheduler`(referring to `sched::scheduler`) to execute processing logic.
- Two IPC channels to receive events from `SchedAgent` and send operations to it.
- A http server to comminicate with `XCLI`.

XServer maintains a global mirror of each process's XQueue status (ready or idle, XPU device ID, process ID, etc.) and trigger the scheduling when receives events.

### Run

You can run xserver by `./output/bin/xserver` in the root directory of XSched after compiling.

The default policy is `HPF`, default port is 50000. You can change them by parameters:

```bash
# Run XServer with default policy and port
./output/bin/xserver
# Run XServer with PUP policy and port 50001
./output/bin/xserver PUP 50001
```
