#!/bin/bash
#
# Launch two HF model benchmark instances with different xsched priorities
# to verify priority-based GPU scheduling on Haiguang DTK.
#
# Prerequisites:
#   1. Start xserver:  cd /path/to/xsched && ./output/bin/xserver HPF 50000 &
#   2. Edit MODEL_PATH below to point to your Qwen3-8B model directory.
#
# Usage:
#   source run_priority_test_hf.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
XSCHED_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DTK_LIB="/opt/dtk-26.04/lib"

# ---- Model path (REQUIRED — edit this) ----
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
HIGH_PRIORITY="${HIGH_PRIORITY:-10}"
LOW_PRIORITY="${LOW_PRIORITY:--10}"
NUM_THREADS="${NUM_THREADS:-4}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"
PROMPT_LEN="${PROMPT_LEN:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

echo "=============================================="
echo "  xsched HF Model Priority Test (HIP/DTK)"
echo "=============================================="
echo "  Model:       ${MODEL_PATH}"
echo "  Threads:     ${NUM_THREADS}"
echo "  Requests:    ${NUM_REQUESTS}"
echo "  Prompt len:  ${PROMPT_LEN}"
echo "  Max tok:     ${MAX_NEW_TOKENS}"
echo "  High prio:   ${HIGH_PRIORITY}"
echo "  Low prio:    ${LOW_PRIORITY}"
echo "=============================================="

if [ ! -d "$MODEL_PATH" ]; then
    echo ""
    echo "[ERRO] Model path not found: ${MODEL_PATH}"
    echo "       Set MODEL_PATH to your Qwen model directory, e.g.:"
    echo "       MODEL_PATH=/data/models/Qwen3-8B source run_priority_test_hf.sh"
    return 1 2>/dev/null || exit 1
fi

# Clean old logs
rm -f test_hf_high.log test_hf_low.log

echo ""
echo "[INFO] Starting HIGH priority instance (priority=${HIGH_PRIORITY})..."
XSCHED_AUTO_XQUEUE_PRIORITY="${HIGH_PRIORITY}" \
    python3 "${SCRIPT_DIR}/benchmark_hf_hip.py" \
    --mode xsched \
    --model "${MODEL_PATH}" \
    --num-threads "${NUM_THREADS}" \
    --num-requests "${NUM_REQUESTS}" \
    --prompt-len "${PROMPT_LEN}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --priority "${HIGH_PRIORITY}" \
    > test_hf_high.log 2>&1 &

HIGH_PID=$!
echo "[INFO] High-priority PID: ${HIGH_PID}"

# Let the first process load the model and start inference
echo "[INFO] Waiting for high-priority instance to initialize..."
sleep 15

echo "[INFO] Starting LOW priority instance (priority=${LOW_PRIORITY})..."
XSCHED_AUTO_XQUEUE_PRIORITY="${LOW_PRIORITY}" \
    python3 "${SCRIPT_DIR}/benchmark_hf_hip.py" \
    --mode xsched \
    --model "${MODEL_PATH}" \
    --num-threads "${NUM_THREADS}" \
    --num-requests "${NUM_REQUESTS}" \
    --prompt-len "${PROMPT_LEN}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --priority "${LOW_PRIORITY}" \
    > test_hf_low.log 2>&1 &

LOW_PID=$!
echo "[INFO] Low-priority PID: ${LOW_PID}"

echo ""
echo "[INFO] Both instances running."
echo "  Logs:  test_hf_high.log / test_hf_low.log"
echo "  Watch: tail -f test_hf_high.log test_hf_low.log"
echo "  Press Ctrl+C to stop early."
echo ""

trap "echo '[INFO] Stopping...'; kill ${HIGH_PID} ${LOW_PID} 2>/dev/null; return 0" INT TERM

# Wait for both
wait ${HIGH_PID} ${LOW_PID} 2>/dev/null || true

echo ""
echo "=============================================="
echo "  Results"
echo "=============================================="

echo ""
echo "--- High Priority (${HIGH_PRIORITY}) ---"
grep -E "Throughput|Avg latency|XSched_Result" test_hf_high.log 2>/dev/null || \
    echo "(check full log: test_hf_high.log)"

echo ""
echo "--- Low Priority (${LOW_PRIORITY}) ---"
grep -E "Throughput|Avg latency|XSched_Result" test_hf_low.log 2>/dev/null || \
    echo "(check full log: test_hf_low.log)"

echo ""
echo "[INFO] Rate history:"
echo "  High: $(grep XSched_Rate:${HIGH_PRIORITY} test_hf_high.log | tail -1)"
echo "  Low:  $(grep XSched_Rate:${LOW_PRIORITY} test_hf_low.log | tail -1)"
