<p align="center">
  <img src="./docs/images/logo.png" alt="xsched project logo" width="128"/>
</p>

# XSched: Preemptive Scheduling for Diverse XPUs

[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://github.com/XpuOS/xsched/blob/main/LICENSE)


---

## Latest News

- [2025/08] We migrated XSched to Windows and completed validation on CUDA, LevelZero, and OpenCL.
- [2025/07] We integrated XSched into [llama.cpp](integration/llama.cpp) and [NVIDIA Triton](integration/triton) to enable priority-based scheduling between multiple inference requests.
- [2025/07] We presented our XSched paper at [OSDI 2025](https://www.usenix.org/conference/osdi25/presentation/shen-weihang) and prepared several interesting demo videos: [Ascend910](https://github.com/user-attachments/assets/bc668f4d-33d9-4492-9900-8c3b10fdd1af), [GV100](https://github.com/user-attachments/assets/dffb821b-92e2-44c5-bd59-a6946b0c4d02), [Least-Laxity-First](https://github.com/user-attachments/assets/885886e1-1920-4fb1-aa2d-50f4f88cf660), and [Active-Window-First](https://github.com/user-attachments/assets/877aeb5f-35b6-4bc1-b553-d76525a8adb3).
- [2025/06] We officially released XSched! Check out our [blog post](docs/xsched-intro-2025-en.md).


---


## Demos
XSched eliminates video stuttering in AI video conference applications on AI PCs

- Hardware: [Intel Core Ultra NPU](https://www.intel.com/content/www/us/en/products/details/processors/core-ultra.html) (i.e., NPU 3720)
- Workloads: fake-background-webcam ([LFBW](https://github.com/fangfufu/Linux-Fake-Background-Webcam)) and speech-to-text ([whisper.cpp](https://github.com/ggml-org/whisper.cpp))
- Scheduling policy: a varient of [least-laxity-first](https://ieeexplore.ieee.org/document/726348) policy

For more details, please see the 2nd case study in our [paper](docs/xsched-osdi25.pdf).

https://github.com/user-attachments/assets/3eb256c3-9107-4d8b-ae8d-0e3ada54aec1


## About

XSched is a preemptive scheduling framework for diverse XPUs (referring to various accelerators, such as GPUs, NPUs, ASICs, and FPGAs) across different brands, generations, and software platforms. XSched provides unified interfaces for scheduling XPU tasks through a preemptible command queue abstraction (XQueue), enabling hardware-agnostic, flexible scheduling policies for various objectives. XSched introduces a multi-level hardware model that helps advanced XPUs achieve optimal scheduling performance while maintaining compatibility with emerging XPUs. The framework is designed to efficiently schedule XPU tasks while remaining transparent to existing XPU-based applications. 


### Features

- **Transparency**: Works with existing applications without code changes.
- **Flexibility**: Supports diverse scheduling policies and accelerator types.
- **Extensibility**: Accommodates new hardware features and software platforms.
- **Integration**: Adapts to both OS-level and system-level multitasking scenarios.
- **Performance**: Delivers high scheduling performance with minimal runtime overhead.


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


## Getting Started

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
# or specify PLATFORM
make PLATFORM = cuda
make PLATFORM = "cuda levelzero opencl" # build multiple platforms at once
```


### Use Cases

#### Transparently Schedule Applications

XSched is designed to be transparent to applications. By setting a few [environment variables](protocol/README.md), you can schedule your application with XSched.
See our example ([Linux](examples/Linux/1_transparent_sched/README.md), [Windows](examples/Windows/1_transparent_sched/README.md)) for transparent scheduling.

#### Linking with XSched for Customized Scheduling

You can also explicitly link with XSched and use XQueue APIs & Hint APIs in your application for more flexibility.
See our examples for more details:

- give hints ([Linux](examples/Linux/2_give_hints/README.md), [Windows](examples/Windows/2_give_hints/README.md))
- intra-process scheduling ([Linux](examples/Linux/3_intra_process_sched/README.md), [Windows](examples/Windows/3_intra_process_sched/README.md))
- manual scheduling ([Linux](examples/Linux/4_manual_sched/README.md), [Windows](examples/Windows/4_manual_sched/cuda/README.md))
- non-transparent scheduling ([Linux](examples/Linux/8_nontransparent_sched/README.md), [Windows](examples/Windows/8_nontransparent_sched/cuda/README.md))

#### More Examples

Check out our [example list](examples/README.md) for more advanced use cases.


## Architecture and Workflow

<img src="/docs/img/xsched-framework.png" alt="XSched framework" width="600" />

XSched consists of four key components: XPU shim (XShim), XPU task preemption module (XPreempt), XPU hardware adapter layer (XAL), and an XScheduler. XShim, XPreempt, and XAL are three dynamically linked libraries that are preloaded into each XPU application process, while XScheduler runs as a centric system service daemon.

- **XShim:** named as `shim` in the code, intercepts XPU driver API calls and redirects commands to the XQueue ①, allowing applications to run on XSched without modifications (transparency).
- **[XPreempt](preempt):** named as `preempt` in the code, implements XQueue interfaces based on the multi-level hardware model ②. Contains an [agent](preempt/src/sched/agent.cpp) that watches the state of XQueue (e.g., ready or idle) and generates scheduling events to notify the XScheduler via IPC ③. Also responsible for applying the scheduling operations (e.g., suspend or resume an XQueue) received from the XScheduler ⑤.
- **XAL:** named as `hal` in the code, implements the multi-level hardware model interfaces by calling XPU driver APIs.
- **[XScheduler](service/server):** named as `xserver` in the code, coordinates all XQueues from different processes, monitors global XQueue status through agent-reported events ③, and invokes the scheduling policy to make decisions when status changes. Decisions are enforced by sending scheduling operations to agents ④. The policy is modular and customizable to suit various workloads.
- **[XCLI](service/cli):** a command-line tool that can monitor XQueue status, change the policy, or give scheduling hints (e.g., priority) ⑥.


## Development Plan

We will continue to support XSched on more OSes and platforms, and improve the performance of XSched. Please stay tuned!

- [x] Integrated into LLM serving systems (e.g., llama.cpp, vLLM)
- [x] Support Windows
- [ ] Support MacOS
- [ ] Install as system daemon


## Contributing

XSched is designed to be extensible and flexible.

We welcome contributions:

- Support more platforms, or a higher preemption level on existing platforms. See [guide](platforms/example/README.md)
- Implement a new scheduling policy. See [guide](sched/README.md)
- Integrate XSched into AI-powered applications. 
- Report or fix issues.


## Citation

If you use XSched for your research, please cite our [paper](docs/xsched-osdi25.pdf):
```bibtex    
@inproceedings{Shen2025xsched,
  title = {{XSched}: Preemptive Scheduling for Diverse {XPU}s},
  author = {Weihang Shen and Mingcong Han and Jialong Liu and Rong Chen and Haibo Chen},
  booktitle = {19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25)},
  year = {2025},
  isbn = {978-1-939133-47-2},
  address = {Boston, MA},
  pages = {671--692},
  url = {https://www.usenix.org/conference/osdi25/presentation/shen-weihang},
  publisher = {USENIX Association},
  month = jul
}
```

The artifacts of XSched is published on [Github](https://github.com/XpuOS/xsched-artifacts) and [Zenodo](https://doi.org/10.5281/zenodo.15327992).

## Contact Us

- For technical questions and feature requests, please submit via GitHub [Issues](https://github.com/XpuOS/xsched/issues)
- For collaborations and partnerships, please reach out at [rongchen@sjtu.edu.cn](mailto:rongchen@sjtu.edu.cn)
