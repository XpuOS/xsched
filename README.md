# XSched: Preemptive Scheduling for Diverse XPUs

[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://github.com/XpuOS/xsched/blob/main/LICENSE)

XSched is a preemptive scheduling framework for XPUs. It provides unified interfaces for scheduling XPU tasks through a preemptible command queue abstraction (XQueue), and proposes a multi-level hardware model that enables advanced XPUs to achieve optimal scheduling performance while maintaining compatibility with emerging XPUs.

## Features

- **Transparency:** Supports existing applications without code change.
- **Flexibility:** Supports multiple scheduling policies and XPUs.
- **Extensibility:** Accommodates new scheduling policies and XPUs easily.
- **Performance:** Delivers high performance (microsecond-scale preemption) with low overhead (< 3%).

## Demos

*comming soon...*

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

## Get Started

### Build and Install XSched

#### Clone

```bash
git clone https://github.com/XpuOS/xsched.git
cd xsched
git submodule update --init --recursive
```

#### Build and Install

```bash
# build XSched only
# XSched will be installed to xsched/output by default
make
# install XSched to a custom path
make INSTALL_PATH=/path/to/install

# build XSched along with platform-specific components (HAL, Shim, etc.)
make cuda # or hip, levelzero, opencl, ascend, cudla, vpi
```

### Core Interfaces

Description coming soon...

### Use cases

#### Transparently Schedule Applications

XSched is designed to be transparent to applications. By setting a few [environment variables](protocol/README.md), you can schedule your application with XSched.
See our [example](examples/1_transparent_sched/README.md) for transparent scheduling.

#### Linking with XSched for Customized Scheduling

You can also explicitly link with XSched and use XQueue APIs & Hint APIs in your application for more flexibility.
See our examples: [give hints](examples/2_give_hints/README.md), [intra-process scheduling](examples/3_intra_process_sched/README.md), and [manual scheduling](examples/4_manual_sched/README.md) for more details.

#### More Examples

Check out our [example list](examples/README.md) for more advanced use cases.

## Architecture and Workflow

<img src="/docs/img/xsched-framework.png" alt="XSched framework" width="600" />

XSched consists of four key components: XPU shim (XShim), XPU task preemption module (XPreempt, named as [`preempt`](preempt) in the code), XPU hardware adapter layer (XAL, named as `hal` in the code), and an [XScheduler](service/server). XShim, XPreempt, and XAL are three dynamically linked libraries that are preloaded into the application process, while XScheduler runs as a centric system service daemon.

- **XShim:** named as `shim` in the code, intercepts XPU driver API calls and redirects commands to the XQueue ①, allowing applications to run on XSched without modifications (transparency).
- **[XPreempt](preempt):** named as `preempt` in the code, implements XQueue interfaces based on the multi-level hardware model ②. Contains an [agent](preempt/src/sched/agent.cpp) that watches the state of XQueue (e.g., ready or idle) and generates scheduling events to notify the XScheduler via IPC ③. Also responsible for applying the scheduling operations (e.g., suspend or resume an XQueue) received from the XScheduler ⑤.
- **XAL:** named as `hal` in the code, implements the multi-level hardware model interfaces by calling XPU driver APIs.
- **[XScheduler](service/server):** named as `xserver` in the code, coordinates all XQueues from different processes, monitors global XQueue status through agent-reported events ③, and invokes the scheduling policy to make decisions when status changes. Decisions are enforced by sending scheduling operations to agents ④. The policy is modular and customizable to suit various workloads.
- **[XCLI](service/cli):** a command-line tool that can monitor XQueue status, change the policy, or give scheduling hints (e.g., priority) ⑥.

## Development Plan

We will continue to support XSched on more OSes and platforms, and improve the performance of XSched. Please stay tuned!

- [ ] Replace cpp-ipc to fix stability issue
- [ ] Support Windows
- [ ] Support MacOS
- [ ] Install as system daemon

## Contributing

XSched is designed to be extensible and flexible.

We welcome contributions:

- Support more platforms, or a higher preemption level on existing platforms. See [guide](platforms/example/README.md)
- Implement a new scheduling policy. See [guide](sched/README.md)
- Report or fix issues.

## Citation

If you use XSched for your research, please cite our [paper](docs/xsched-osdi25.pdf):
```bibtex    
@inproceedings{Shen2025xsched,
  title = {{XSched}: Preemptive Scheduling for Diverse {XPU}s},
  author = {Weihang Shen and Mingcong Han and Jialong Liu and Rong Chen and Haibo Chen},
  booktitle = {19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25)},
  year = {2025},
  address = {Boston, MA},
  url = {https://www.usenix.org/conference/osdi25/presentation/shen-weihang},
  publisher = {USENIX Association},
  month = jul
}
```

The artifacts of XSched is published on [Github](https://github.com/XpuOS/xsched-artifacts) and [Zenodo](https://doi.org/10.5281/zenodo.15327992).
