#!/bin/bash
#
# Launch two model servers with different xsched UTILIZATION for
# UP (Utilization Partition) policy on Haiguang DTK.
#
# Prerequisites:
#   1. Start xserver with UP:  ./output/bin/xserver UP 50000 &
#   2. Qwen3-8B model at MODEL_PATH.
#
# Usage:
#   MODEL_PATH=/models/Qwen3-8B source run_servers_up.sh
#   start_high    (terminal 1 — 80% GPU time)
#   start_low     (terminal 2 — 20% GPU time)
#   bench_high    (terminal 3)
#   bench_low     (terminal 4)
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
HIGH_UTIL="${HIGH_UTIL:-80}"   # 80% GPU time for high utilization
LOW_UTIL="${LOW_UTIL:-20}"     # 20% GPU time for low utilization
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"
CONCURRENCY="${CONCURRENCY:-4}"

echo "=============================================="
echo "  xsched UP Policy Test (Utilization Partition)"
echo "=============================================="
echo "  Model:        ${MODEL_PATH}"
echo "  High: port ${HIGH_PORT}, GPU#${HIGH_GPU}, util ${HIGH_UTIL}%"
echo "  Low:  port ${LOW_PORT}, GPU#${LOW_GPU}, util ${LOW_UTIL}%"
echo "=============================================="
echo ""

if [ ! -d "$MODEL_PATH" ]; then
    echo "[WARN] Model path not found: ${MODEL_PATH}"
    echo "       Set MODEL_PATH before sourcing."
    echo ""
fi

start_high() {
    echo "[INFO] Starting HIGH utilization server (util=${HIGH_UTIL}%, port ${HIGH_PORT}, GPU#${HIGH_GPU})..."
    XSCHED_AUTO_XQUEUE_UTILIZATION="${HIGH_UTIL}" \
        python3 "${SCRIPT_DIR}/model_server.py" \
        --model "${MODEL_PATH}" \
        --port "${HIGH_PORT}" \
        --gpu "${HIGH_GPU}" \
        --priority 0
}

start_low() {
    echo "[INFO] Starting LOW utilization server (util=${LOW_UTIL}%, port ${LOW_PORT}, GPU#${LOW_GPU})..."
    XSCHED_AUTO_XQUEUE_UTILIZATION="${LOW_UTIL}" \
        python3 "${SCRIPT_DIR}/model_server.py" \
        --model "${MODEL_PATH}" \
        --port "${LOW_PORT}" \
        --gpu "${LOW_GPU}" \
        --priority 0
}

bench_high() {
    local reqs="${1:-${NUM_REQUESTS}}"
    local conc="${2:-${CONCURRENCY}}"
    python3 "${SCRIPT_DIR}/bench_client.py" \
        --url "http://localhost:${HIGH_PORT}/generate" \
        --concurrency "${conc}" \
        --requests "${reqs}" \
        --max-new-tokens "${MAX_NEW_TOKENS}"
}

bench_low() {
    local reqs="${1:-${NUM_REQUESTS}}"
    local conc="${2:-${CONCURRENCY}}"
    python3 "${SCRIPT_DIR}/bench_client.py" \
        --url "http://localhost:${LOW_PORT}/generate" \
        --concurrency "${conc}" \
        --requests "${reqs}" \
        --max-new-tokens "${MAX_NEW_TOKENS}"
}

echo "[INFO] Environment ready. Available commands:"
echo "  start_high           — HIGH util server (${HIGH_UTIL}% GPU, port ${HIGH_PORT})"
echo "  start_low            — LOW util server  (${LOW_UTIL}% GPU, port ${LOW_PORT})"
echo "  bench_high [N] [C]   — bench HIGH (N requests, C concurrency)"
echo "  bench_low [N] [C]    — bench LOW"
echo ""
echo "[INFO] Override: HIGH_UTIL=70 LOW_UTIL=30 source run_servers_up.sh"
echo ""
