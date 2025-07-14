# Use XSched to Schedule Tasks inside Llama.cpp Inference Server

This example demonstrates how XSched can be integrated into Llama.cpp Inference Server to enable priority-based scheduling of one-model multiple-instance tasks.

**The source code of this example is presented in this [repo](https://github.com/XpuOS/llama.cpp/tree/xsched).**

## Basic Idea

Although Llama.cpp server supports multiple-users and parallel decoding, it is unable to set the priority of one request.

With only a few lines of code change, we modify the backend of llama.cpp `ggml-backend` to create an XQueue for each CUDA stream and inherit the priority of the model using the stream.

The vanilla Llama.cpp Server provides service in an one-model one-instance manner and implements parallel decoding by continuous batching technique, so there is no room to schedule in a priority-based way. We modify Llama.cpp Server to an one-model multiple-instance way, which means it shares one copy of LLM's weights and handles inference tasks independently among instances.

We use local scheduler and highest priority first policy to schedule these XQueues, similar to the [Intra-Process Scheduling Example](../3_intra_process_sched/README.md).

With XSched, the inference tasks of the model using the higher-priority XQueues can preempt the lower-priority ones, so that their latencies can be significantly reduced and their throughputs can be significantly increased.

## Download the LLM

We take Deepseek-R1 as an example here.

```bash
cd /models
wget https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF/resolve/main/DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf
```

## Build Modified Llama.cpp Server

First, build XSched.

```bash
cd /xsched
make cuda
```

Then, build the modified Llama.cpp Server.

```bash
cd /
git clone [git@github.com:XpuOS/llama.cpp.git](https://github.com/XpuOS/llama.cpp.git)
cd llama.cpp
git checkout -b xsched origin/xsched
cmake -B build -DGGML_CUDA=on -DCMAKE_PREFIX_PATH=/xsched/output/lib
cmake --build build -- -j$(nproc)
```

Now, start the server.

```bash
cd /llama.cpp
LD_LIBRARY_PATH=/xsched/output/lib:$LD_LIBRARY_PATH XSCHED_POLICY=HPF ./build/bin/llama-server -m /models/DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf -ngl 99 -c 4096 -np 2
```

Use the script to chat with the server.

```bash
# normal priority
./llama.cpp/tools/server/chat.sh 0 
# higher priority
./llama.cpp/tools/server/chat.sh 1 
```
