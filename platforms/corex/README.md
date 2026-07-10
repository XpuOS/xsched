# CoreX Platform Support for XSched

The [CoreX (天数智芯)](https://www.iluvatar.com/) adapter supports transparent Level-1 scheduling for CoreX GPU applications through CoreX's CUDA-compatible APIs.

CoreX SDK exposes both CUDA runtime-style APIs (`cuda*`) and CUDA driver-style APIs (`cu*`). CoreX frameworks may mix the two families in one process, so the shim intercepts both, including stream creation, synchronization, events, memcpy, and kernels.

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
    <td align="center">CoreX CUDA-compatible APIs</td>
    <td align="center">CoreX GPUs</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">🔘</td>
    <td align="center">❌</td>
  </tr>
</table>

## Build

```bash
make corex
```

The build installs `libhalcorex.so` and `libshimcorex.so` under `output/lib`.

## Run

Make the CoreX runtime libraries and XSched libraries visible, then preload the CoreX shim:

```bash
export LD_LIBRARY_PATH=/path/to/xsched/output/lib:/path/to/corex/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/path/to/xsched/output/lib/libshimcorex.so

export XSCHED_SCHEDULER=GLB
export XSCHED_AUTO_XQUEUE=ON
export XSCHED_AUTO_XQUEUE_LEVEL=1
export XSCHED_AUTO_XQUEUE_PRIORITY=0 # change as you need
```

If the CoreX driver or runtime libraries are not found automatically, override them with `XSCHED_COREX_LIB` and `XSCHED_COREX_RT_LIB`.

Start `xserver` when using the global scheduler:

```bash
/path/to/xsched/output/bin/xserver HPF 50000
```

Then run the CoreX application normally in another shell with the same environment variables.
