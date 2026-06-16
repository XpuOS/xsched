# XSched Support for vLLM

At the moment, XSched support vLLM though tranparent schduling method. You can refer to the [doc](../8_nontransparent_sched/README.md) for more details about transparent scheduling.

vLLM uses CUDA Graphs to accelerate inference, and XSched has supported CUDA Graphs since v1.2.0.
In XSched, a CUDA Graph is treated as a single command.
Therefore, preempting it requires Level-3 preemption, namely running command preemption in our paper, which is currently supported only by the TSG-based implementation.

To enable preemption of CUDA Graphs, set the environment variables:

```bash
export XSCHED_AUTO_XQUEUE=ON      # automatically create XQueues for each CUDA stream
export XSCHED_AUTO_XQUEUE_LEVEL=3 # use Level-3 for auto-created XQueues
export XSCHED_CUDA_LV3_IMPL=TSG   # use TSG-based implementation for Level-3 preemption
```

Note that the TSG implementation suspends and resumes the entire process (or more precisely, the entire CUDA context) as a single unit.
Therefore, if you wish to achieve fine-grained scheduling within a process, you need to disable CUDA Graphs so that vLLM submits commands as individual kernels.

To disable CUDA Graphs in vLLM:

```bash
export VLLM_USE_CUDA_GRAPH=0
export VLLM_NO_CUDA_GRAPH=1
```

or

```python
import os
os.environ["VLLM_USE_CUDA_GRAPH"] = "0"
os.environ["VLLM_NO_CUDA_GRAPH"]  = "1"

# them init vLLM
```
