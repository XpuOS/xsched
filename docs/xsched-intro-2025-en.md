## XSched: A Unified, Efficient, Flexible Multi-Task Scheduler for Heterogeneous Computing 

[English](xsched-intro-2025-en.md) | [中文](xsched-intro-2025-zh.md)

Our work [**XSched**](https://www.usenix.org/conference/osdi25/presentation/shen-weihang), published on OSDI'25, has been [open-sourced](https://github.com/XpuOS/xsched) on GitHub! 🚀

We are in the era of "Computing Unit Explosion": From cloud to edge, heterogeneous accelerators (XPUs) such as GPUs, NPUs, ASICs, and FPGAs from vendors like NVIDIA, [AMD](https://www.amd.com/en/products/graphics/radeon-ai.html), [Intel](https://www.intel.com/content/www/us/en/products/details/embedded-processors/core-ultra.html), [Ascend](https://www.hiascend.com/), [Cambricon](https://www.cambricon.com/), and [MetaX](https://www.metax-tech.com/) are being widely deployed. The trend toward intelligent applications implies that XPUs will become as essential as CPUs for all workloads. However, the management and scheduling mechanisms for these valuable computing resources largely remain in the "Prehistoric Era" of exclusive resource allocation. The built-in hardware schedulers in current XPUs employ simple and rigid policies, typically limited to basic non-preemptive scheduling like FCFS (First-Come, First-Served), making them inadequate for diverse requirements such as real-time performance, priority and fairness. This is particularly critical for rapidly evolving Chinese GPUs and AI chips, where computational resource competition is an urgent challenge, while feature richness remains difficult to achieve. On the other hand, most software scheduling research focuses on GPUs with mature ecosystems and advanced functionalities, delivering powerful yet highly customized optimizations that are hard to generalize across different XPU types, brands or even generations. The goal of our project is straightforward: **We aim to make XPU scheduling support multi-task sharing and feature uniformity, efficiency, flexibility simultaneously, just like CPU scheduling now.**

To tame these vastly diverse, rapidly evolving, and constantly innovating computing hardware, we avoid crafting hardware-specific "perfect wheels" for every new accelerator. Instead, we adopt an operating system perspective to address two fundamental challenges in hardware management: system abstraction and hardware modele.


### 1️⃣ System Abstraction: Preemptive Command Queue 💡

Inspired by the CPU-scheduling abstraction 'Thread' in operating system, we propose a unified XPU-scheduling abstraction: Preemptive Command Queue (XQueue). XQueue provides a set of unified, concise interfaces, such as submit, wait, suspend, resume and etc. and fully decouples high-level scheduling **policy** and low-level hardware **mechanisms**. For application programmers, they will work with a unified abstraction of computing hardware, focusing model algorithms and business logic without considering about how tasks are scheduled and utilize resources on hardware. For system programmers, they can code a scheduling policy once and deploy it on XPUs from different vendors and generations, even enabling across-XPU scheduling, which significantly reduces development and maintenance costs. Based on the abstraction of XQueue, Xsched has implemented diverse scheduling [policies](https://github.com/XpuOS/xsched/blob/main/sched/README.md) and deployed on heterogeneous computing platforms.

<div align="center">
<img src="/docs/img/xqueue-abstraction.png" alt="XQueue Abstraction" width="600" />
</div>

### 2️⃣ Hardware Model: Multi-Level Hardware Model Based on Scheduling Capabilities 🛠️

We fully recognize the significant and evolving differences in hardware capabilities. Unlike the management of CPU and memory, which adopts a singular hardware module to describe hardware capabilities, we propose multi-level hardware model and design a three levels of task preemption. This model acts as a 'Rosetta Stone', enabling XSched to communicate efficiently with any XPU.

- **Level 1 (Pending Command Preemption)**: it is adaptive to almost every computing hardware of different architectures, including GPU, NPU, ASIC, and FPGA. With general methods, simple development and minimal porting costs (directly reuse under the same architecture), it ensures broad generality.
- **Level 2 (In-flight Command Preemption)**: it leverages more powerful hardware task management capabilities to preempt submitted but not executed command. This greatly reduces the preemption latency and improves system responsiveness and fairness. Currently this level is supported by most GPU and NPU, and can be used in closed-source drivers and application frameworks.
- **Level 3 (Running Command Preemption)**: it leverages the most powerful hardware task management capabilities to preempt running command. This level can achieve micro-seconds preemption latency to satisfy real-time requirement and safety. Hardware with capabilities of task switching and interrupts can support this level.

<div align="center">
<img src="/docs/img/preemption-levels.png" alt="Three preemption levels for XPU scheduling" width="600" />
</div>


**Unified System Abstraction** and **Multi-level Hardware Model** not only solve current compatibility challenges, but also direct a clear path for the computing hardware's functional advancement and ecosystem development by well-defined capability levels. For more detailed techniques，we invite you to read our [paper](https://ipads.se.sjtu.edu.cn/_media/publications/xsched-osdi25.pdf).


## XSched's Recent Advancement
Currently, XSched has supported over ten mas-production XPU devices from NVIDIA, AMD, Intel, Ascend and Xilinx. It is the first time to achieve multi-task preemptive scheduling on NPU and ASIC computing platforms and is also the first time to achieve the highest level of preemptive scheduling on closed-sourced GPU platforms. The open-sourced XSched is compatible with mainstream heterogeneous computing platforms including CUDA, HIP, OneAPI, AscendCL, OpenCL (see [XPU Support Matrix](https://github.com/XpuOS/xsched/blob/main/README.md)), which means you can directly deploy XSched on laptops, PCs and servers to experience the improvements brought by multi-task scheduling.

#### Not only an academic prototype XSched is, it has also demonstrated application potential in many real-world scenarios.

- **☁️ Cloud Cost Optimization**: In a multi-tenant cloud scenario hosting multiple containers on a single GPU, **without modifying application codes** (transparent to users) and with almost **no influence** to high-priority users (<%1 overhead), XSched fully utilizes the expensive hardware and harvests 2.74× more GPU resources than SOTA system, [TGS](https://github.com/pkusys/TGS)[^1].
- **🤖 Reduce AI Service Latency**: With just **a dozen lines of code**, XSched can be integrated into [NVIDIA Triton](https://github.com/triton-inference-server/server), a production-level inference service server, and reduces the tail latency of high priority requests by 1.41×. Compared to NVIDIA GPU-specific SOTA scheduling system [Paella](https://github.com/eniac/paella)[^2], XSched achieves 1.30× lower tail latency under heavy workloads.
- **💻 Fluent Experience for Edge Device**: For AI video conference application running on [Intel Core Ultra NPU](https://www.intel.cn/content/www/cn/zh/products/details/processors/core-ultra.html), XSched implements a Modified Least Laxity First (MLLF[^3]) XPU scheduling policy to ensure the **real-time response of foreground tasks** (e.g., speech-to-text) and significantly reduce the frame latency of background fake-background application (LFBW) by 9.26×, which eliminates stuttering and enhances user experience.

We have prepared system demonstration videos on different platforms to show XSched's powerful scheduling abilities in multi-task scenarios.

https://github.com/user-attachments/assets/dffb821b-92e2-44c5-bd59-a6946b0c4d02

https://github.com/user-attachments/assets/bc668f4d-33d9-4492-9900-8c3b10fdd1af

https://github.com/user-attachments/assets/885886e1-1920-4fb1-aa2d-50f4f88cf660

https://github.com/user-attachments/assets/877aeb5f-35b6-4bc1-b553-d76525a8adb3


[Here](https://github.com/XpuOS/xsched/tree/main/examples) is more examples to use XSched.

---

XSched represents our crucial stride towards solving XPU multi-task scheduling challenges and changing the usage of XPU in a "General" way instead of a "Specialized" way in the perspective of abstracting and managing heterogeneous computing hardware. We eagerly anticipate collaborating with academic and industry peers to jointly build the foundation for next-generation computing systems.

**Welcome to Star, Fork, Try, and Contribute!** 🙏

🔗 [Github](https://github.com/XpuOS/xsched) 📄 [paper](https://ipads.se.sjtu.edu.cn/_media/publications/xsched-osdi25.pdf)


[^1]: Transparent GPU Sharing in Container Clouds for Deep Learning Workloads. Bingyang Wu, Zili Zhang, Zhihao Bai, Xuanzhe Liu, and Xin Jin. Symposium on Networked Systems Design and Implementation (NSDI), 2023. https://www.usenix.org/conference/nsdi23/presentation/wu 
[^2]: Paella: Low-latency Model Serving with Software-defined GPU Scheduling. Kelvin K. W. Ng, Henri Maxime Demoulin, and Vincent Liu. Symposium on Operating Systems Principles (SOSP), 2023. https://dl.acm.org/doi/10.1145/3600006.3613163
[^3]: A Modified Least-Laxity-First Scheduling Algorithm for Real-time Tasks. Sung-Heun Oh and Seung-Min Yang. International Conference on Real-Time Computing Systems and Applications (RTCSA), 1998.  https://ieeexplore.ieee.org/abstract/document/726348
