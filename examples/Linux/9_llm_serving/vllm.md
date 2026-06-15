# XSched Support for vLLM

At the moment, XSched support vLLM though tranparent schduling method. You can refer to the [doc](../8_nontransparent_sched/README.md) for more details about transparent scheduling.

Since XSched does not support CUDA graph yet, some environment variables need to be set to disable CUDA graph in vLLM. You can set the environment variable `VLLM_USE_CUDA_GRAPH=0` to disable CUDA graph in vLLM or add the following code snippet to your program before initializing vLLM:

```python
import os
os.environ["VLLM_USE_CUDA_GRAPH"] = "0"
os.environ["VLLM_NO_CUDA_GRAPH"]  = "1"
```