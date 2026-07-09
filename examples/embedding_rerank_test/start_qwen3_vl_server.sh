#!/bin/bash

# 参数 1: [HPF / OFF]
MODE=${1:-HPF}

# --- 请根据实际模型路径修改 ---
EMB_MODEL="/root/models/Qwen3-VL-Embedding-2B"
RERANK_MODEL="/root/models/Qwen3-VL-Reranker-2B"
CLIP_MODEL="/root/models/chinese-clip-vit-base-patch16" # 请确认此路径

# --- 环境路径设置 ---
XSCHED_ROOT="/root/workspace/vllm_xsched/vllm-xsched/output"
SHIM_LIB="${XSCHED_ROOT}/lib/libshimcuda.so"
XSERVER_BIN="${XSCHED_ROOT}/bin/xserver"

# 1. 停止旧进程
pkill -9 python3
pkill -9 xserver
sleep 1

# 2. 启动 XServer (如果是 HPF 模式)
if [ "$MODE" == "HPF" ]; then
    echo "🚀 Starting XServer in HPF Mode..."
    $XSERVER_BIN HPF 50000 &
    sleep 2
    export LD_LIBRARY_PATH=$(dirname $SHIM_LIB):$LD_LIBRARY_PATH
    export LD_PRELOAD=$SHIM_LIB
    export XSCHED_SCHEDULER="GLB"
    export XSCHED_AUTO_XQUEUE="ON"
    export XSCHED_AUTO_XQUEUE_THRESHOLD=1
    export XSCHED_AUTO_XQUEUE_BATCH_SIZE=1
else
    echo "⚠️ Starting in NATIVE CUDA mode (OFF)..."
    unset LD_PRELOAD
fi

# 3. 启动服务
launch_service() {
    local task=$1
    local port=$2
    local prio=$3
    local model_path=$4
    local script="examples/embedding_rerank_test/api_server_qwen3_vl.py"
    
    # 针对 clip 任务使用专门的独立脚本
    if [ "$task" == "clip" ]; then
        script="examples/embedding_rerank_test/api_server_clip.py"
    fi
    
    echo "🔥 Launching $task on port $port (Prio: $prio) using $script..."
    export XSCHED_AUTO_XQUEUE_PRIORITY=$prio
    
    python3 "$script" \
        --task "$task" \
        --port "$port" \
        --model_path "$model_path" \
        --shim-path "$SHIM_LIB" \
        --default-priority "$prio" &
}

# 分配端口和优先级
# HPF 模式下：Embedding 和 CLIP 设为普通优先级(1)，Rerank 设为 VIP 优先级(10)
if [ "$MODE" == "HPF" ]; then
    launch_service "embedding" 8891 1 "$EMB_MODEL"
    launch_service "rerank"    8892 10 "$RERANK_MODEL"
    launch_service "clip"      8893 1 "$CLIP_MODEL"
else
    # OFF 模式（原生 CUDA）
    launch_service "embedding" 8891 10 "$EMB_MODEL"
    launch_service "rerank"    8892 10 "$RERANK_MODEL"
    launch_service "clip"      8893 10 "$CLIP_MODEL"
fi

echo "⏳ Waiting for models to load..."
wait
