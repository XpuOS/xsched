## XSched: 为异构计算打造通用、高效、灵活的多任务调度器

[English](xsched-intro-2025-en.md) | [中文](xsched-intro-2025-zh.md)

我们发表在OSDI'25上的工作[**XSched**](https://www.usenix.org/conference/osdi25/presentation/shen-weihang)，目前已在GitHub[开源](https://github.com/XpuOS/xsched)！🚀

当前我们正处在一个“计算单元大爆炸”的时代：从云端到边缘，NVIDIA、[AMD](https://www.amd.com/en/products/graphics/radeon-ai.html)、[Intel](https://www.intel.com/content/www/us/en/products/details/embedded-processors/core-ultra.html)、[昇腾](https://www.hiascend.com/)、[寒武纪](https://www.cambricon.com/)、[沐曦](https://www.metax-tech.com/)等厂商的GPU、NPU、ASIC、FPGA等各类异构加速器（XPU）被广泛部署。应用的智能化趋势意味着XPU将会如同CPU一样成为所有应用的必须品。然而，对于这些宝贵算力的管理和调度机制还普遍停留在个别应用独占资源的“史前时代”。当前XPU内置的硬件调度器策略单一且固化，通常仅支持FCFS（先来先服务）等最基本的非抢占式调度，难以满足实时性、优先性、公平性等多样化场景需求。尤其是处于快速发展中的国产GPU和AI芯片，算力竞争是当务之急，而丰富功能则难以兼顾。另一方面，软件调度技术的研究大多聚焦在拥有丰富生态和先进功能的GPU上，提供强大却也是高度定制的优化设计，难以应用到不同类型、不同品牌、甚至是不同世代的XPU上。我们项目的出发点很简单：**希望让XPU调度像今天的CPU调度一样，在支持丰富功能的同时拥有通用性、灵活性、可编程性。**

为了驯服这些有着巨大差异、快速演进、不断创新的算力硬件，我们没有尝试为每一款新硬件重复造最适合当下的轮子，而是从操作系统的视角探索管理硬件的两大核心问题：系统抽象与硬件模型

### 1️⃣ 系统抽象：可抢占的命令队列 💡

我们借鉴操作系统中CPU调度抽象“线程”（Thread），提出了一个统一的XPU调度抽象：可抢占命令队列（XQueue）。它提供了一套统一、简洁的接口（如，提交、等待、挂起、恢复等），将上层的调度**策略**与底层的硬件**机制**彻底解耦。这意味着，应用程序员将面向统一的算力硬件抽象，聚焦模型算法和业务逻辑的开发，无需考虑任务在硬件上如何调度和使用资源。同时，系统程序员可以编写一次调度策略，就能让它在不同厂商、不同代际的XPU上通用，甚至实现跨XPU的调度，从而大大降低了开发和维护成本。基于XQueue抽象，XSched实现了多种[调度策略](https://github.com/XpuOS/xsched/blob/main/sched/README.md)，能够使用在各类异构算力平台上。

<div align="center">
<img src="/docs/img/xqueue-abstraction.png" alt="XQueue Abstraction" width="600" />
</div>

### 2️⃣ 硬件模型：调度能力分级建模 🛠️

我们深知XPU的硬件能力千差万别且处于快速演进中。为此，我们没有像管理CPU和内存那样采用单一的硬件模型来描述硬件功能，提出了多级硬件建模的思想，并针对XPU任务调度的特征，设计了一个三级任务抢占能力模型，像一块“罗塞塔石碑”，让XSched能与任何XPU高效“对话”。

- **Level 1（待提交命令抢占级）**：适用于不同架构的几乎所有算力硬件（包括：GPU、NPU、ASIC和FPGA），方法通用开发简单，移植成本极低（同架构下能够直接复用），保证了最广泛的通用性。
- **Level 2（已提交未执行命令抢占级）**：利用更强的硬件任务管控能力，抢占已提交但未执行的命令，能够大幅降低任务抢占延迟，提升系统交互性和公平性。当前大部分GPU和NPU都可以支持该级别，且能够支持闭源的驱动和应用框架。
- **Level 3（执行中命令抢占级）**：利用最强的硬件任务管理能力，中断正在执行中的命令，能够实现微秒级的超低延迟抢占，满足应用的实时和安全需求。提供了任务切换或者中断能力的硬件能够支持该级别。

<div align="center">
<img src="/docs/img/preemption-levels.png" alt="Three preemption levels for XPU scheduling" width="600" />
</div>


**统一系统抽象**和**多级硬件建模**不仅解决了当下的兼容性难题，更通过定义清晰的能力层级，为算力硬件的功能演进和生态建设指明了方向。更多技术细节，欢迎大家阅读我们的[论文](https://ipads.se.sjtu.edu.cn/_media/publications/xsched-osdi25.pdf)。


## XSched最新进展

目前，XSched已经支持了来自NVIDIA、AMD、Intel、昇腾、Xilinx的十余款量产XPU设备，首次在NPU和ASIC算力平台上实现了多任务抢占式调度，也是第一次在闭源GPU平台上实现最高级别的抢占式调度。XSched开源实现兼容CUDA、HIP、oneAPI、AscendCL、OpenCL等主流异构计算平台（详见[当前硬件支持列表](https://github.com/XpuOS/xsched/blob/main/README.md)），意味着你可以直接在笔记本、PC、服务器上直接部署XSched，体验多任务调度带来的变化。


#### XSched不仅是一个学术原型，它在多个真实场景中都展示了潜在的应用价值：

- **☁️ 为云服务商降本增效**：在GPU多容器混合部署场景下，XSched能在**不修改应用代码**（对客户透明）且**几乎不影响**高优先级客户性能（<1% 开销）的同时，相比SOTA系统[TGS](https://github.com/pkusys/TGS)[^1]，多压榨出了2.74倍的GPU利用率，实现昂贵硬件的效率最大化。
- **🤖 为智能服务削减延迟**：只需**约10行代码**，XSched就能集成到工业级推理服务框架[NVIDIA Triton](https://github.com/triton-inference-server/server)中，并为其提供低延迟的**多任务抢占式调度能力**，将高优先级请求的尾延迟降低1.41倍。相比NVIDIA GPU定制的SOTA调度系统[Paella](https://github.com/eniac/paella)[^2]，能在高负载下取得1.3倍的尾延迟降低。
- **💻 为终端带来流畅体验**：对于运行在 [Intel Core Ultra NPU](https://www.intel.cn/content/www/cn/zh/products/details/processors/core-ultra.html) 上的智能视频会议应用，XSched实现了一种基于改进最低松弛度优先(MLLF)的XPU调度策略[^3]，保障语音转文字等**前台任务的实时响应**，同时让视频背景虚化等后台特效的帧处理**延迟显著降低**9.26 倍，消除卡顿并提升用户体验。

我们在不同硬件平台上准备了系统演示视频，展示XSched在多任务场景下的强大调度效果！

https://github.com/user-attachments/assets/dffb821b-92e2-44c5-bd59-a6946b0c4d02

https://github.com/user-attachments/assets/bc668f4d-33d9-4492-9900-8c3b10fdd1af

https://github.com/user-attachments/assets/885886e1-1920-4fb1-aa2d-50f4f88cf660


更多XSched的使用示例详见[应用示例]（https://github.com/XpuOS/xsched/tree/main/examples）

---

XSched是我们在抽象和管理异构算力硬件方向上，解决XPU多任务调度挑战，并从“专用”走向“通用”的重要一步。我们非常期待与学术界和工业界的同行们交流、合作，共同构建下一代计算系统的基石。

**欢迎大家 Star、Fork、试用和共建！** 🙏

🔗 [Github](https://github.com/XpuOS/xsched) 📄 [论文直达](https://ipads.se.sjtu.edu.cn/_media/publications/xsched-osdi25.pdf)


[^1]: Transparent GPU Sharing in Container Clouds for Deep Learning Workloads. Bingyang Wu, Zili Zhang, Zhihao Bai, Xuanzhe Liu, and Xin Jin. Symposium on Networked Systems Design and Implementation (NSDI), 2023. https://www.usenix.org/conference/nsdi23/presentation/wu 
[^2]: Paella: Low-latency Model Serving with Software-defined GPU Scheduling. Kelvin K. W. Ng, Henri Maxime Demoulin, and Vincent Liu. Symposium on Operating Systems Principles (SOSP), 2023. https://dl.acm.org/doi/10.1145/3600006.3613163
[^3]: A Modified Least-Laxity-First Scheduling Algorithm for Real-time Tasks. Sung-Heun Oh and Seung-Min Yang. International Conference on Real-Time Computing Systems and Applications (RTCSA), 1998.  https://ieeexplore.ieee.org/abstract/document/726348
