# Platform Support for XSched

## Structure

Each platform adapter is split into two parts:

- `hal`: the XPU adapter layer (XAL). It implements XSched's `HwCommand` and `HwQueue` abstractions on top of the platform driver/runtime APIs.
- `shim`: the transparent interception layer (XShim). It intercepts application API calls and redirects asynchronous commands to XQueues when XSched is enabled.

Some platforms also provide language bindings or platform-specific tests.

## XAL

The XAL library converts native XPU work, such as a kernel launch, memcpy, or operator execution, into `HwCommand` objects submitted to an `XQueue`.

Common `HwCommand` interfaces:

| Interface               | Description                                      |
| ----------------------- | ------------------------------------------------ |
| Enqueue()               | Launch the native command through the platform API |
| Synchronize()           | Wait for the command when supported             |
| Synchronizable()        | Report whether the command supports waiting     |
| EnableSynchronization() | Attach the event/fence needed for waiting       |

`HwQueue` is an abstraction of real device queue (such as `zeCommandQueue` and `zeImmediateCommandList` in LevelZero, `CUstream` in CUDA). We need to implement different functions to support different preemption level.

<table>
  <tr>
    <th align="center">Preemption Level</th>
    <th align="center">Interface</th>
    <th align="center">Description</th>
  </tr>
  <tr>
    <td align="center" rowspan="2">Level-1</td>
    <td align="center">Launch(HwCommand)</td>
    <td align="left">Launch a HwCommand by calling HwCommand->Enqueue()</td>
  </tr>
  <tr>
    <td align="center">Synchronize()</td>
    <td align="left">Wait for all commands in the HwQueue to complete</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">Level-2</td>
    <td align="center">Deactivate()</td>
    <td align="left">Deactivate the HwQueue to prevent all its commands from being selected for execution</td>
  </tr>
  <tr>
    <td align="center">Reactivate()</td>
    <td align="left">Reactivate the HwQueue to allow all its commands to be selected for execution</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">Level-3</td>
    <td align="center">Interrupt()</td>
    <td align="left">Interrupt the running command of the HwQueue</td>
  </tr>
  <tr>
    <td align="center">Restore()</td>
    <td align="left">Restore the interrupted command of the HwQueue</td>
  </tr>
</table>

Only Level-1 is required for a usable platform adapter.

## XShim

The XShim library preserves application transparency.
It intercepts the platform API calls that create queues, submit asynchronous work, record or wait for events, synchronize, and release resources whose lifetime can affect queued work.

- `intercept.cpp` exports intercepted platform symbols and redirects them.
- `shim.cpp` implements XSched-specific handling for those symbols.

## Supported Platforms

| Name      | Usage                         |
| --------- | ----------------------------- |
| CUDA      | [README](cuda/README.md)      |
| CoreX     | [README](corex/README.md)     |
| HIP       | [README](hip/README.md)       |
| LevelZero | [README](levelzero/README.md) |
| OpenCL    | [README](opencl/README.md)    |
| AscendCL  | [README](ascend/README.md)    |
| cuDLA     | [README](cudla/README.md)     |
| VPI       | [README](vpi/README.md)       |

If you are interested in supporting XSched on a new platform, please refer to our [example and guide](example/README.md).
