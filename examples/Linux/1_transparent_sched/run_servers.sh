#!/bin/bash
#
# Launch two model servers with different xsched priorities for side-by-side
# GPU scheduling comparison on Haiguang DTK.
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
#     source run_servers.sh  # sets up env
#     start_high             # starts server on port 8080, priority 10
#
#   Terminal 2 — start LOW priority server (port 8081):
#     cd examples/Linux/1_transparent_sched
#     source run_servers.sh
#     start_low              # starts server on port 8081, priority -10
#
#   Terminal 3 — benchmark HIGH priority:
#     cd examples/Linux/1_transparent_sched
#     source run_servers.sh
#     bench_high             # sends 32 requests to port 8080
#
#   Terminal 4 — benchmark LOW priority:
#     cd examples/Linux/1_transparent_sched
#     source run_servers.sh
#     bench_low              # sends 32 requests to port 8081
#
#   Both servers print per-request latency in real time in their terminals.
#   The bench clients print aggregate throughput/latency.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XSCHED_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DTK_LIB="/opt/dtk-26.04/lib"

# ---- Model path (REQUIRED) ----
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-8B}"

# ---- xsched environment ----
export XSCHED_SCHEDULER="GLB"
export XSCHED_AUTO_XQUEUE="ON"
export XSCHED_AUTO_XQUEUE_THRESHOLD=16
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=8
export XSCHED_HIP_LIB="${DTK_LIB}/libamdhip64.so"
export XSCHED_HIP_COMGR_LIB="${DTK_LIB}/libamdcomgr.so"

# ---- library paths ----
export LD_LIBRARY_PATH="${XSCHED_DIR}/output/lib:${DTK_LIB}:${LD_LIBRARY_PATH}"
export LD_PRELOAD="${XSCHED_DIR}/output/lib/libshimhip.so"

# ---- test parameters ----
HIGH_PORT="${HIGH_PORT:-8080}"
LOW_PORT="${LOW_PORT:-8081}"
HIGH_GPU="${HIGH_GPU:-0}"
LOW_GPU="${LOW_GPU:-1}"
HIGH_PRIORITY="${HIGH_PRIORITY:-10}"
LOW_PRIORITY="${LOW_PRIORITY:--10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"
CONCURRENCY="${CONCURRENCY:-4}"

echo "=============================================="
echo "  xsched Model Server Priority Test"
echo "=============================================="
echo "  Model:        ${MODEL_PATH}"
echo "  High: port ${HIGH_PORT}, priority ${HIGH_PRIORITY}"
echo "  Low:  port ${LOW_PORT}, priority ${LOW_PRIORITY}"
echo "=============================================="
echo ""

# Check model path
if [ ! -d "$MODEL_PATH" ]; then
    echo "[WARN] Model path not found: ${MODEL_PATH}"
    echo "       Set MODEL_PATH before sourcing, e.g.:"
    echo "       MODEL_PATH=/data/models/Qwen3-8B source run_servers.sh"
    echo ""
fi

# ---- Convenience functions ----

start_high() {
    echo "[INFO] Starting HIGH priority server on port ${HIGH_PORT} (GPU#${HIGH_GPU})..."
    XSCHED_AUTO_XQUEUE_PRIORITY="${HIGH_PRIORITY}" \
        python3 "${SCRIPT_DIR}/model_server.py" \
        --model "${MODEL_PATH}" \
        --port "${HIGH_PORT}" \
        --priority "${HIGH_PRIORITY}" \
        --gpu "${HIGH_GPU}"
}

start_low() {
    echo "[INFO] Starting LOW priority server on port ${LOW_PORT} (GPU#${LOW_GPU})..."
    XSCHED_AUTO_XQUEUE_PRIORITY="${LOW_PRIORITY}" \
        python3 "${SCRIPT_DIR}/model_server.py" \
        --model "${MODEL_PATH}" \
        --port "${LOW_PORT}" \
        --priority "${LOW_PRIORITY}" \
        --gpu "${LOW_GPU}"
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

echo "[INFO] Environment ready. Available commands:"
echo "  start_high   — start HIGH priority server (port ${HIGH_PORT})"
echo "  start_low    — start LOW priority server  (port ${LOW_PORT})"
echo "  bench_high   — benchmark HIGH priority server"
echo "  bench_low    — benchmark LOW priority server"
echo ""
echo "[INFO] GPU memory note: two Qwen3-8B instances need ~32 GiB VRAM."
echo "       If your GPU has less memory, use a smaller model or set"
echo "       CONCURRENCY=1 NUM_REQUESTS=1 to test sequentially."
echo ""
