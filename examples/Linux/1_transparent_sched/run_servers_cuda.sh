#!/bin/bash
#
# Launch two model servers with different xsched priorities for side-by-side
# GPU scheduling comparison on NVIDIA CUDA.
#
# Prerequisites:
#   1. Start xserver:  ./output/bin/xserver HPF 50000 &
#   2. Qwen3-8B model downloaded to MODEL_PATH.
#
# ============================================================
#  How to use (open 4 terminals)
# ============================================================
#
#   Terminal 1 — start HIGH priority server (port 8080):
#     cd examples/Linux/1_transparent_sched
#     MODEL_PATH=/data/models/Qwen3-8B source run_servers_cuda.sh
#     start_high
#
#   Terminal 2 — start LOW priority server (port 8081):
#     cd examples/Linux/1_transparent_sched
#     MODEL_PATH=/data/models/Qwen3-8B source run_servers_cuda.sh
#     start_low
#
#   Terminal 3 — benchmark HIGH priority:
#     cd examples/Linux/1_transparent_sched
#     source run_servers_cuda.sh
#     bench_high
#
#   Terminal 4 — benchmark LOW priority:
#     cd examples/Linux/1_transparent_sched
#     source run_servers_cuda.sh
#     bench_low
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XSCHED_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CUDA_LIB="/usr/local/cuda/lib64"

# ---- Model path (REQUIRED) ----
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-8B}"

# ---- xsched environment (CUDA/NVIDIA) ----
export XSCHED_SCHEDULER="GLB"
export XSCHED_AUTO_XQUEUE="ON"
export XSCHED_AUTO_XQUEUE_THRESHOLD=16
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=8
export XSCHED_CUDA_LIB="${CUDA_LIB}/libcuda.so"

# ---- library paths ----
export LD_LIBRARY_PATH="${XSCHED_DIR}/output/lib:${CUDA_LIB}:${LD_LIBRARY_PATH}"
export LD_PRELOAD="${XSCHED_DIR}/output/lib/libshimcuda.so"

# ---- test parameters ----
HIGH_PORT="${HIGH_PORT:-8080}"
LOW_PORT="${LOW_PORT:-8081}"
HIGH_PRIORITY="${HIGH_PRIORITY:-10}"
LOW_PRIORITY="${LOW_PRIORITY:--10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"
CONCURRENCY="${CONCURRENCY:-4}"

echo "=============================================="
echo "  xsched Model Server Priority Test (CUDA)"
echo "=============================================="
echo "  Model:        ${MODEL_PATH}"
echo "  High: port ${HIGH_PORT}, priority ${HIGH_PRIORITY}"
echo "  Low:  port ${LOW_PORT}, priority ${LOW_PRIORITY}"
echo "=============================================="
echo ""

if [ ! -d "$MODEL_PATH" ]; then
    echo "[WARN] Model path not found: ${MODEL_PATH}"
    echo "       Set MODEL_PATH before sourcing."
    echo ""
fi

start_high() {
    echo "[INFO] Starting HIGH priority server on port ${HIGH_PORT}..."
    XSCHED_AUTO_XQUEUE_PRIORITY="${HIGH_PRIORITY}" \
        python3 "${SCRIPT_DIR}/model_server.py" \
        --model "${MODEL_PATH}" \
        --port "${HIGH_PORT}" \
        --priority "${HIGH_PRIORITY}"
}

start_low() {
    echo "[INFO] Starting LOW priority server on port ${LOW_PORT}..."
    XSCHED_AUTO_XQUEUE_PRIORITY="${LOW_PRIORITY}" \
        python3 "${SCRIPT_DIR}/model_server.py" \
        --model "${MODEL_PATH}" \
        --port "${LOW_PORT}" \
        --priority "${LOW_PRIORITY}"
}

bench_high() {
    python3 "${SCRIPT_DIR}/bench_client.py" \
        --url "http://localhost:${HIGH_PORT}/generate" \
        --concurrency "${CONCURRENCY}" \
        --requests "${NUM_REQUESTS}" \
        --max-new-tokens "${MAX_NEW_TOKENS}"
}

bench_low() {
    python3 "${SCRIPT_DIR}/bench_client.py" \
        --url "http://localhost:${LOW_PORT}/generate" \
        --concurrency "${CONCURRENCY}" \
        --requests "${NUM_REQUESTS}" \
        --max-new-tokens "${MAX_NEW_TOKENS}"
}

echo "[INFO] Environment ready. Commands:"
echo "  start_high / start_low   — start servers"
echo "  bench_high / bench_low   — run clients"
echo ""
echo "[INFO] GPU memory: two Qwen3-8B need ~32 GiB VRAM."
echo "       Use a smaller model if VRAM is insufficient."
echo ""
