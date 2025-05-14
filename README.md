# XSched

[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://github.com/XpuOS/xsched/blob/main/LICENSE)

XSched is a preemptive scheduling framework for XPUs. It provides unified interfaces for scheduling XPU tasks through a preemptible command queue abstraction (XQueue), and proposes a multi-level hardware model that enables mature, advanced XPUs to achieve optimal scheduling performance.

## Features

- **Transparency:** Don't require any code changes to existing applications.
- **Flexibility:** Support multiple scheduling policies and XPUs.
- **Extensibility:** Easy to adapt new scheduling policies and XPUs.
- **Performance:** Achieve high performance with low overhead.

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

## Demo

*comming soon...*

## Get Started

### Build and Install XSched

#### 1. Clone the repository

```bash
git clone https://github.com/XpuOS/xsched.git
cd xsched
```

#### 2. Install dependencies

```bash
git submodule update --init --recursive
```

#### 3. Build XSched by need

```bash
# Build for CUDA
make cuda
# Build for HIP
make hip
# Build for LevelZero
make levelzero
# Build for OpenCL
make opencl
# Build for AscendCL
make ascend
# Build for cuDLA
make cudla
# Build for VPI
make vpi
```

### Transparently Schedule Applications

XSched is designed to be transparent to applications. By setting a few [environment variables](protocol/README.md), you can schedule your application with XSched.

See our [example](examples/1_transparency/README.md) for transparent scheduling.

### Linking with XSched for Customized Scheduling

You can also explicitly link with XSched and use XQueue APIs & Hint APIs in your application for more flexibility.

See our [example](examples/2_link_xsched/README.md) for linking with XSched.

## Architecture and Workflow

<img src="/docs/img/xsched-framework.png" alt="XSched framework" width="600" />

1. The XShim library changes the workflow by intercepting XPU driver API calls and redirecting commands to the XQueue.
2. Commands submitted to the XQueue are buffered and launched to the XPU at a proper time.
3. The XPreempt library contains an agent that watches the state of XQueue (e.g., ready or idle) and generates scheduling events to notify the scheduler via IPC.
4. The XSched daemon enforces the decisions of the policy by sending scheduling operations.
5. The agent applies the scheduling operations (e.g., suspend or resume an XQueue) received from the scheduler.
6. XCLI is provided to users to change the policy and give scheduling hints like priorities.

## TODOs

We will continue to support XSched on more OSes and platforms, and improve the performance of XSched. Please stay tuned!

- [ ] Replace cpp-ipc to fix stability issue
- [ ] Install as system daemon
- [ ] Support MacOS
- [ ] Support Windows

## Contributing

XSched is designed to be extensible and flexible.

We welcome contributions:

- Support more platforms, or a higher preemption level on existing platforms. See [guide](platforms/example/README.md)
- Implement a new scheduling policy. See [guide](sched/README.md)
- Report or fix issues.

## Paper and Citation

*Coming soon.*
