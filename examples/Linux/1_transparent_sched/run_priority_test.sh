#!/bin/bash
#
# xsched priority scheduling test — runs two PyTorch model instances
# with different priorities and measures throughput.
#
# Prerequisites:
#   1. Start xserver first:  cd /path/to/xsched && ./output/bin/xserver HPF 50000
#   2. Source this script's exports for the test environment.
#
# Usage:
#   source run_priority_test.sh
#   # or:  bash run_priority_test.sh
#
# The script starts two background processes:
#   - High priority (priority=10):  logs to test_high_priority.log
#   - Low priority  (priority=-10): logs to test_low_priority.log
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
XSCHED_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DTK_LIB="/opt/dtk-26.04/lib"

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
MODEL="${MODEL:-resnet50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ITERS="${ITERS:-500}"
WARMUP="${WARMUP:-20}"

HIGH_PRIORITY="${HIGH_PRIORITY:-10}"
LOW_PRIORITY="${LOW_PRIORITY:--10}"

echo "=============================================="
echo " xsched Priority Scheduling Test"
echo "=============================================="
echo " Model:       ${MODEL}"
echo " Batch size:  ${BATCH_SIZE}"
echo " Iterations:  ${ITERS}"
echo " High prio:   ${HIGH_PRIORITY}"
echo " Low prio:    ${LOW_PRIORITY}"
echo "=============================================="
echo ""

# Clean up old logs
rm -f test_high_priority.log test_low_priority.log

# ---- Launch high-priority instance ----
echo "[INFO] Starting high-priority instance (priority=${HIGH_PRIORITY})..."
XSCHED_AUTO_XQUEUE_PRIORITY=${HIGH_PRIORITY} \
    python3 "${SCRIPT_DIR}/test_model.py" \
    --model "${MODEL}" \
    --batch-size "${BATCH_SIZE}" \
    --iters "${ITERS}" \
    --warmup "${WARMUP}" \
    > test_high_priority.log 2>&1 &

HIGH_PID=$!
echo "[INFO] High-priority PID: ${HIGH_PID}"

# Small delay to let the first process initialize
sleep 2

# ---- Launch low-priority instance ----
echo "[INFO] Starting low-priority instance (priority=${LOW_PRIORITY})..."
XSCHED_AUTO_XQUEUE_PRIORITY=${LOW_PRIORITY} \
    python3 "${SCRIPT_DIR}/test_model.py" \
    --model "${MODEL}" \
    --batch-size "${BATCH_SIZE}" \
    --iters "${ITERS}" \
    --warmup "${WARMUP}" \
    > test_low_priority.log 2>&1 &

LOW_PID=$!
echo "[INFO] Low-priority PID: ${LOW_PID}"

echo ""
echo "[INFO] Both instances running. Waiting for completion..."
echo "[INFO] Logs: test_high_priority.log / test_low_priority.log"
echo "[INFO] Press Ctrl+C to kill both instances early."
echo ""

# Trap Ctrl+C to kill both processes
trap "echo '[INFO] Stopping...'; kill ${HIGH_PID} ${LOW_PID} 2>/dev/null; exit 0" INT TERM

# Wait for both to finish
wait ${HIGH_PID} ${LOW_PID} 2>/dev/null || true

echo ""
echo "=============================================="
echo " Results"
echo "=============================================="

# Extract results
echo ""
echo "--- High Priority (${HIGH_PRIORITY}) ---"
grep "RESULT" test_high_priority.log 2>/dev/null || echo "(no RESULT line found — check log)"

echo ""
echo "--- Low Priority (${LOW_PRIORITY}) ---"
grep "RESULT" test_low_priority.log 2>/dev/null || echo "(no RESULT line found — check log)"

echo ""
echo "Full logs: test_high_priority.log / test_low_priority.log"
