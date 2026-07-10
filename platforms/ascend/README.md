# AscendCL Platform Support for XSched

The AscendCL adapter supports transparent Level-1 scheduling for Ascend NPU applications. It intercepts AscendCL events, memcpy, synchronization, and single-operator execution APIs, then submits asynchronous work to XSched XQueues.

## Support

<table>
  <tr>
    <th align="center">Platform</th>
    <th align="center">XPU</th>
    <th align="center">Shim</th>
    <th align="center">Level-1</th>
    <th align="center">Level-2</th>
    <th align="center">Level-3</th>
  </tr>
  <tr>
    <td align="center">AscendCL</td>
    <td align="center">Ascend NPUs</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">🔘</td>
    <td align="center">❌</td>
  </tr>
</table>

## Build

```bash
make ascend
```

The build installs `libhalascend.so` and `libshimascend.so` under `output/lib`.

## Run

Source the Ascend toolkit environment first, then preload the shim:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export LD_LIBRARY_PATH=/path/to/xsched/output/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/path/to/xsched/output/lib/libshimascend.so

export XSCHED_SCHEDULER=GLB
export XSCHED_AUTO_XQUEUE=ON
export XSCHED_AUTO_XQUEUE_LEVEL=1
export XSCHED_AUTO_XQUEUE_PRIORITY=0 # change as you need
```

Start `xserver` when using the global scheduler:

```bash
/path/to/xsched/output/bin/xserver HPF 50000
```

Then run the AscendCL or framework application normally in another shell with the same environment variables.
