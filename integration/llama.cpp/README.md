# XSched Integration for llama.cpp

This demonstrates how XSched can be integrated into llama.cpp to enable priority-based scheduling between multiple inference requests.

## Basic Idea

We modify the backend of llama.cpp `ggml-backend` to create an XQueue for a CUDA stream and add an API to set its priority.
We also modify the `server` example to a one-model multiple-instance manner and set the priority of the XQueue of each instance.
Then, we use local scheduler and highest priority first policy to schedule these XQueues.

## Usage

### Apply Integration Patch

```bash
# commit id: 73e53dc834c0a2336cd104473af6897197b96277
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 73e53dc834c0a2336cd104473af6897197b96277
git apply <xsched_dir>/integration/llama.cpp/llamacpp-xsched-73e53dc.patch
```

or, use our forked repo

```bash
git clone https://github.com/XpuOS/llama.cpp.git -b xsched
```

### Build XSched
```bash
cd <xsched_dir>
make cuda
```

### Build llama.cpp Server

```bash
cd <llama.cpp_dir>
cmake -B build -DGGML_CUDA=on -DCMAKE_PREFIX_PATH=<xsched_dir>/output/lib
cmake --build build -- -j$(nproc)
```

### Run Server

```bash
cd <llama.cpp_dir>
export LD_LIBRARY_PATH=<xsched_dir>/output/lib:$LD_LIBRARY_PATH
export XSCHED_POLICY=HPF
./build/bin/llama-server -m <model_path> -ngl 9999 -c 4096 -np 2
```

## Example

See [llama.cpp example](../../examples/8_llama.cpp) for concrete details.
