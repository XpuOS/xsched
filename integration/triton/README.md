# XSched Integration for NVIDIA Triton Inference Server

This demonstrates how XSched can be integrated into NVIDIA Triton Inference Server to enable priority-based scheduling of multi-model inference tasks.

## Basic Idea

Triton Server allows users to set the priority of each serving model.
We modify the TensorRT-Backend of the Triton Server to create an XQueue for each CUDA stream and inherit the priority of the model using the stream.
Then, we use local scheduler and highest priority first policy to schedule these XQueues.
With XSched, the inference tasks of the model using the higher-priority XQueues can preempt the lower-priority ones (Triton does not support task preemption currently), so that their latencies can be significantly reduced.

## Usage

### Apply Integration Patch

```bash
# commit id: 7f94a8ee1daab23046ef4d689bd56411101f207c
git clone https://github.com/triton-inference-server/tensorrt_backend -b r22.06
cd tensorrt_backend
git apply <xsched_dir>/integration/triton/trt-xsched-2206.patch
```

### Build XSched
```bash
# Setup Triton Server container and mount the xsched and tensorrt_backend directory (replace `<xsched_dir>` and `<tensorrt_backend_dir>` with your own location)
docker run --privileged -itd --name xsched-triton-server --gpus all --net=host -v <xsched_dir>:/xsched -v <tensorrt_backend_dir>:/root/tensorrt_backend nvcr.io/nvidia/tritonserver:22.06-py3 bash
docker exec -it xsched-triton-server bash

# Install Miniconda in the container
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init --all

# Install dependencies, we need cmake 3.17+ and rapidjson
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda install -y cmake rapidjson

# Build XSched
cd /xsched
make cuda
```

### Build TensorRT-Backend

```bash
docker exec -it xsched-triton-server bash
source ~/miniconda3/bin/activate

# Build TensorRT-Backend
cd /root/tensorrt_backend

mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_COMMON_REPO_TAG=r22.06 -DTRITON_CORE_REPO_TAG=r22.06 -DTRITON_BACKEND_REPO_TAG=r22.06 -DXSched_DIR=/xsched/output/lib/cmake/XSched ..
make -j$(nproc)
make install
```

### Run Triton Server

In `config.pbtxt`, you can set the priority of each model.

```
optimization {
  priority: PRIORITY_MAX
}
```

or,

```
optimization {
  priority: PRIORITY_MIN
}
```

Then start the server
```bash
export LD_LIBRARY_PATH=/xsched/output/lib:$LD_LIBRARY_PATH
export XSCHED_POLICY=HPF
tritonserver --backend-directory ./install/backends --model-repository=<model_repo_dir> --strict-model-config false
```

## Example

See [inference serving example](../../examples/5_infer_serving) for concrete details.
