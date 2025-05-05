# XSched

XSched is a preemptive scheduling framework for XPUs.

## Features


## Demo


## Get Started

### Build and Install XSched

### Transparently Schedule Applications

XSched is designed to be transparent to applications. By setting a few environment variables, you can schedule your application with XSched. See our [example](examples/1_transparency) for transparent scheduling.

### Linking with XSched for Customized Scheduling

You can also explicitly link with XSched and use XQueue APIs & Hint APIs in your application for more flexibility. See our [example](examples/2_link_xsched) for linking with XSched.

## XPU Support Matrix

✅ supported and implemented

❌ not supported

🔘 not yet implemented

🚧 implementation within progress

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
    <td align="center" rowspan="4"><a href="platforms/cuda">CUDA</a></td>
    <td align="center">NVIDIA Ampere GPUs (sm86)</td>
    <td align="center" rowspan="4">✅</td>
    <td align="center" rowspan="4">✅</td>
    <td align="center">🚧</td>
    <td align="center">🚧</td>
  </tr>
  <tr>
    <td align="center">NVIDIA Volta GPUs (sm70)</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
  </tr>
  <tr>
    <td align="center">NVIDIA Kepler GPUs (sm35)</td>
    <td align="center">✅</td>
    <td align="center">❌</td>
  </tr>
  <tr>
    <td align="center">Other NVIDIA GPUs</td>
    <td align="center">🔘</td>
    <td align="center">🔘</td>
  </tr>
  <tr>
    <td align="center" rowspan="1"><a href="platforms/hip">HIP</a></td>
    <td align="center">AMD GPUs</td>
    <td align="center" rowspan="1">✅</td>
    <td align="center" rowspan="1">✅</td>
    <td align="center">🔘</td>
    <td align="center">🔘</td>
  </tr>
  <tr>
    <td align="center" rowspan="2"><a href="platforms/levelzero">LevelZero</a></td>
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
  <tr>
    <td align="center" rowspan="4"><a href="platforms/opencl">OpenCL</a></td>
    <td align="center">NVIDIA GPUs</td>
    <td align="center" rowspan="4">✅</td>
    <td align="center" rowspan="4">✅</td>
    <td align="center">🔘</td>
    <td align="center">🔘</td>
  </tr>
  <tr>
    <td align="center">AMD GPUs</td>
    <td align="center">🔘</td>
    <td align="center">🔘</td>
  </tr>
  <tr>
    <td align="center">Intel GPUs</td>
    <td align="center">🔘</td>
    <td align="center">🔘</td>
  </tr>
  <tr>
    <td align="center">Xillinx FPGAs</td>
    <td align="center">🔘</td>
    <td align="center">❌</td>
  </tr>
  <tr>
    <td align="center" rowspan="1"><a href="platforms/ascend">AscendCL</a></td>
    <td align="center">Ascend NPUs</td>
    <td align="center" rowspan="1">✅</td>
    <td align="center" rowspan="1">✅</td>
    <td align="center">🔘</td>
    <td align="center">❌</td>
  </tr>
  <tr>
    <td align="center" rowspan="1"><a href="platforms/cudla">cuDLA</a></td>
    <td align="center">NVIDIA DLA</td>
    <td align="center" rowspan="1">✅</td>
    <td align="center" rowspan="1">✅</td>
    <td align="center">❌</td>
    <td align="center">❌</td>
  </tr>
  <tr>
    <td align="center" rowspan="2"><a href="platforms/vpi">VPI</a></td>
    <td align="center">NVIDIA OFA</td>
    <td align="center" rowspan="2">✅</td>
    <td align="center" rowspan="2">✅</td>
    <td align="center">❌</td>
    <td align="center">❌</td>
  </tr>
  <tr>
    <td align="center">NVIDIA PVA</td>
    <td align="center">❌</td>
    <td align="center">❌</td>
  </tr>
</table>



## Architecture and Workflow


## TODOs

We will continue to support XSched on more OSes and platforms, and improve the performance of XSched. Please stay tuned!

- [ ] Replace cpp-ipc to fix stability issue
- [ ] Install as system daemon
- [ ] Support MacOS
- [ ] Support Windows

## Contributing

XSched is designed to be extensible and flexible.

We welcome contributions:

- Support more platforms, or a higher preemption level on existing platforms. See [guide](platforms/README.md)
- Implement a new scheduling policy. See [guide](sched/README.md)
- Report or fix issues.

## Paper and Citation


