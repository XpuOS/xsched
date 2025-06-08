# Use XSched to Schedule Tasks inside one Inference Server Process

This example demonstrates how XSched can be integrated into Triton Inference Server to enable priority-based scheduling of multi-model inference tasks. This example is one of the experiments (Figure 15a) in the [XSched paper](../../docs/xsched-osdi25.pdf) and is included in the [XSched Artifacts](https://github.com/XpuOS/xsched-artifacts/tree/main/cuda/fig13a).

The example requires three docker images:
- TensorRT docker image: `nvcr.io/nvidia/tensorrt:22.06-py3`
- Triton Server docker image: `nvcr.io/nvidia/tritonserver:22.06-py3`
- Triton Client docker image: `nvcr.io/nvidia/tritonserver:22.06-py3-sdk`

Steps:
  - [Download the BERT model from NGC](#download-the-bert-model-from-ngc)
  - [Build TensorRT model](#build-tensorrt-model)
  - [Build TensorRT-Backend for Triton Server](#build-tensorrt-backend-for-triton-server)
  - [Run Triton Client (Standalone, Triton, Triton + Priority Config)](#run-triton-client-standalone-triton-triton--priority-config)
  - [Build Triton Server with XSched \& Run Triton Client (XSched)](#build-triton-server-with-xsched--run-triton-client-xsched)
  - [Plot the results](#plot-the-results)

## Download the BERT model from NGC

```bash
bash BERT/download.sh
```

## Build TensorRT model

Convert the TensorFlow model to TensorRT executable model.

```bash
# Use TensorRT 8.5 docker image and mount the xsched directory (replace `<xsched_dir>` with your own location)
docker run --privileged -it --rm --gpus all --net=host -v <xsched_dir>:/xsched nvcr.io/nvidia/tensorrt:22.06-py3 bash

# Install dependencies
python3 -m pip install --upgrade pip
pip install onnx==1.16.0 tensorflow==2.13.0

# Build TensorRT model
cd /xsched/examples/5_infer_serving

python python/builder.py -m BERT/bert_large_tf1_ckpt_mode-qa_ds-squad11_insize-384_19.03.1/model.ckpt -b 1 -s 384 -c BERT/bert_large_tf1_ckpt_mode-qa_ds-squad11_insize-384_19.03.1/ --fp16 -o model.plan

# Copy the model to the model repository
cp model.plan /xsched/examples/5_infer_serving/model-repo/bert-high/1/
cp model.plan /xsched/examples/5_infer_serving/model-repo/bert-low/1/
cp model.plan /xsched/examples/5_infer_serving/model-repo/bert-norm/1/
```

## Build TensorRT-Backend for Triton Server

```bash
# Setup Triton Server container and mount the xsched directory (replace `<xsched_dir>` with your own location)
docker run --privileged -itd --name xsched-triton-server --gpus all --net=host -v <xsched_dir>:/xsched nvcr.io/nvidia/tritonserver:22.06-py3 bash
docker exec -it xsched-triton-server bash

# Install Miniconda in the container
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init --all

# Install dependencies
conda install -y cmake rapidjson
```

```bash
# Build TensorRT-Backend for Triton Server

cd /root

# commit id: 7f94a8ee1daab23046ef4d689bd56411101f207c
git clone https://github.com/triton-inference-server/tensorrt_backend -b r22.06
cd tensorrt_backend

mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_COMMON_REPO_TAG=r22.06 -DTRITON_CORE_REPO_TAG=r22.06 -DTRITON_BACKEND_REPO_TAG=r22.06 ..
make -j$(nproc)
make install
```

```bash
# Run Triton Server
tritonserver --backend-directory ./install/backends --model-repository=/xsched/examples/5_infer_serving/model-repo/ --strict-model-config false

# You can see the output of the server in the container terminal, like this:
# +----------+---------------------------------------------------+--------------------------------------------------------------------------------------------------+
# | Backend  | Path                                              | Config                                                                                           |
# +----------+---------------------------------------------------+--------------------------------------------------------------------------------------------------+
# | tensorrt | ./install/backends/tensorrt/libtriton_tensorrt.so | {"cmdline":{"auto-complete-config":"true","min-compute-capability":"6.000000","backend-directory |
# |          |                                                   | ":"./install/backends","default-max-batch-size":"4"}}                                            |
# +----------+---------------------------------------------------+--------------------------------------------------------------------------------------------------+

# I0608 15:29:19.380337 2941 server.cc:626] 
# +-----------+---------+--------+
# | Model     | Version | Status |
# +-----------+---------+--------+
# | bert-high | 1       | READY  |
# | bert-low  | 1       | READY  |
# | bert-norm | 1       | READY  |
# +-----------+---------+--------+

# I0608 15:29:19.391999 2941 metrics.cc:650] Collecting metrics for GPU 0: Quadro GV100
# I0608 15:29:19.392079 2941 tritonserver.cc:2159] 
# +----------------------------------+------------------------------------------------------------------------------------------------------------------------------+
# | Option                           | Value                                                                                                                        |
# +----------------------------------+------------------------------------------------------------------------------------------------------------------------------+
# | server_id                        | triton                                                                                                                       |
# | server_version                   | 2.23.0                                                                                                                       |
# | server_extensions                | classification sequence model_repository model_repository(unload_dependents) schedule_policy model_configuration system_shar |
# |                                  | ed_memory cuda_shared_memory binary_tensor_data statistics trace                                                             |
# | model_repository_path[0]         | /xsched/examples/5_infer_serving/model-repo/                                                                                 |
# | model_control_mode               | MODE_NONE                                                                                                                    |
# | strict_model_config              | 0                                                                                                                            |
# | rate_limit                       | OFF                                                                                                                          |
# | pinned_memory_pool_byte_size     | 268435456                                                                                                                    |
# | cuda_memory_pool_byte_size{0}    | 67108864                                                                                                                     |
# | response_cache_byte_size         | 0                                                                                                                            |
# | min_supported_compute_capability | 6.0                                                                                                                          |
# | strict_readiness                 | 1                                                                                                                            |
# | exit_timeout                     | 30                                                                                                                           |
# +----------------------------------+------------------------------------------------------------------------------------------------------------------------------+

# I0608 15:29:19.396067 2941 grpc_server.cc:4587] Started GRPCInferenceService at 0.0.0.0:8001
# I0608 15:29:19.396875 2941 http_server.cc:3303] Started HTTPService at 0.0.0.0:8000
# I0608 15:29:19.437883 2941 http_server.cc:178] Started Metrics Service at 0.0.0.0:8002
```

## Run Triton Client (Standalone, Triton, Triton + Priority Config)

```bash
# mount the xsched directory (replace `<xsched_dir>` with your own location)
# Evaluate the standalone performance
docker run --privileged -it --rm --gpus all --net=host -v <xsched_dir>:/xsched nvcr.io/nvidia/tritonserver:22.06-py3-sdk bash -c "cd /xsched/examples/5_infer_serving; bash client/run.sh standalone"

# Evaluate the performance with triton server
docker run --privileged -it --rm --gpus all --net=host -v <xsched_dir>:/xsched nvcr.io/nvidia/tritonserver:22.06-py3-sdk bash -c "cd /xsched/examples/5_infer_serving; bash client/run.sh triton"

# Evaluate the performance with triton + priority config
docker run --privileged -it --rm --gpus all --net=host -v <xsched_dir>:/xsched nvcr.io/nvidia/tritonserver:22.06-py3-sdk bash -c "cd /xsched/examples/5_infer_serving; bash client/run.sh triton-p"
```

Results are saved in `<xsched_dir>/examples/5_infer_serving/results/raw`.

## Build Triton Server with XSched & Run Triton Client (XSched)

Let's go back to the Triton Server container (xsched-triton-server) and build XSched (remember to stop the triton server by typing `Ctrl-C`).

```bash
# Build XSched

cd /xsched
make cuda


# Integrate XSched into Triton Server TensorRT-Backend
cd /root/tensorrt_backend

git apply /xsched/examples/5_infer_serving/server/triton-backend-xsched.patch

# Rebuild the TensorRT-Backend
cd build
cmake -DXSched_DIR=/xsched/output/lib/cmake/XSched ..
make
make install
```

Now, let's start the Triton Server again with XSched.

```bash
export LD_LIBRARY_PATH=/xsched/output/lib:$LD_LIBRARY_PATH
export XSCHED_POLICY=HPF
tritonserver --backend-directory ./install/backends --model-repository=/xsched/examples/5_infer_serving/model-repo/ --strict-model-config false
```

Finally, let's run the evaluation script again.

```bash
docker run --privileged -it --rm --gpus all --net=host -v <xsched_dir>:/xsched nvcr.io/nvidia/tritonserver:22.06-py3-sdk bash -c "cd /xsched/examples/5_infer_serving; bash client/run.sh xsched"
```

## Plot the results

Raw results are under `<xsched_dir>/examples/5_infer_serving/results/raw`. To plot the figure in the paper, please install gnuplot and epstopdf.

```bash
sudo apt update
sudo apt install gnuplot texlive-font-utils

# process data and plot
python3 <xsched_dir>/examples/5_infer_serving/results/scripts/plot.py
```

Figure 13a are under `<xsched_dir>/examples/5_infer_serving/results/plot`.
