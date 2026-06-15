# LevelZero Platform Support for XSched

## Supported preemption level

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
    <td align="center" rowspan="2">LevelZero</td>
    <td align="center">Intel GPUs</td>
    <td align="center" rowspan="2">✅</td>
    <td align="center" rowspan="2">✅</td>
    <td align="center">🔘</td>
    <td align="center">🔘</td>
  </tr>
  <tr>
    <td align="center">Intel Integrated NPUs</td>
    <td align="center">✅</td>
    <td align="center">❌</td>
  </tr>
</table>

## Build

Build the Level Zero platform adapter:

```bash
make levelzero
```

## Run

Start `xserver` when using the global scheduler:

```bash
export LD_LIBRARY_PATH=/path/to/xsched/output/lib:$LD_LIBRARY_PATH
/path/to/xsched/output/bin/xserver HPF 50000
```

Set the transparent XQueue environment before running the application:

```bash
export LD_LIBRARY_PATH=/path/to/xsched/output/lib:$LD_LIBRARY_PATH
export XSCHED_SCHEDULER=GLB
export XSCHED_AUTO_XQUEUE=ON
export XSCHED_AUTO_XQUEUE_LEVEL=1
export XSCHED_AUTO_XQUEUE_PRIORITY=0 # change as you need
```

Then run the application normally.

```bash
# Take test as an example
cd platforms/levelzero/test
python3 npu.py
```

See [transparent scheduling](../../examples/Linux/1_transparent_sched/README.md) for the general workflow.
