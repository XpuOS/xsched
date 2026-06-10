#!/usr/bin/env python3
"""
Lightweight HF model HTTP server for xsched priority scheduling test (HIP/DTK).

Each request runs inference on a non-default HIP stream — xsched intercepts
hipStreamCreate via LD_PRELOAD and creates an xqueue with this process's priority.

Usage:
    # Terminal 1 — high priority server
    XSCHED_AUTO_XQUEUE_PRIORITY=10 python model_server.py --port 8080 --priority 10

    # Terminal 2 — low priority server
    XSCHED_AUTO_XQUEUE_PRIORITY=-10 python model_server.py --port 8081 --priority -10

    # Terminal 3/4 — send requests
    curl -s -X POST http://localhost:8080/generate \
      -H "Content-Type: application/json" \
      -d '{"prompt": "你好，请介绍一下深度学习。", "max_new_tokens": 256}'

    # Or use a benchmark client:
    python bench_client.py --url http://localhost:8080/generate --concurrency 4 --requests 32
"""

import os
import sys
import time
import json
import argparse
import threading
import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

# Global model/tokenizer, set during startup
MODEL = None
TOKENIZER = None
DEVICE = None
PRIORITY = 0

# Metrics tracking
metrics_lock = threading.Lock()
request_count = [0]
total_tokens = [0]
total_time = [0.0]
start_time = None


def print_status(prefix: str, **kwargs):
    """Print a status line with priority tag and timing info."""
    elapsed = time.time() - start_time if start_time else 0
    parts = " ".join(f"{k}={v}" for k, v in kwargs.items())
    msg = f"[prio={PRIORITY: >4}] [{elapsed:7.1f}s] [{request_count[0]:4d} req] {prefix} | {parts}"
    print(msg, flush=True)


@app.route("/generate", methods=["POST"])
def generate():
    """Generate text from a prompt. Expects JSON: {prompt, max_new_tokens?, ...}"""
    global request_count, total_tokens, total_time

    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    max_new_tokens = data.get("max_new_tokens", 128)
    temperature = data.get("temperature", 1.0)
    do_sample = data.get("do_sample", False)

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    # Create a non-default stream — xsched intercepts hipStreamCreate
    stream = torch.cuda.Stream()

    t0 = time.perf_counter()

    with torch.cuda.stream(stream):
        inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = MODEL.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=TOKENIZER.eos_token_id,
            )

    stream.synchronize()
    latency = time.perf_counter() - t0

    output_tokens = outputs.shape[1] - input_len
    generated_text = TOKENIZER.decode(outputs[0][input_len:], skip_special_tokens=True)

    # Update metrics
    with metrics_lock:
        request_count[0] += 1
        total_tokens[0] += output_tokens
        total_time[0] += latency

    throughput = output_tokens / latency if latency > 0 else 0
    avg_throughput = total_tokens[0] / total_time[0] if total_time[0] > 0 else 0

    print_status("OK",
                 latency=f"{latency*1000:.0f}ms",
                 tokens=f"{input_len}→{output_tokens}",
                 tok_s=f"{throughput:.0f}",
                 avg_tok_s=f"{avg_throughput:.0f}")

    return jsonify({
        "generated_text": generated_text,
        "input_tokens": input_len,
        "output_tokens": output_tokens,
        "latency_ms": round(latency * 1000, 1),
        "tokens_per_sec": round(throughput, 1),
    })


@app.route("/stats", methods=["GET"])
def stats():
    """Return current metrics."""
    with metrics_lock:
        avg_latency = (total_time[0] / request_count[0] * 1000) if request_count[0] > 0 else 0
        avg_throughput = total_tokens[0] / total_time[0] if total_time[0] > 0 else 0

    return jsonify({
        "priority": PRIORITY,
        "requests": request_count[0],
        "total_tokens": total_tokens[0],
        "avg_latency_ms": round(avg_latency, 1),
        "avg_tokens_per_sec": round(avg_throughput, 1),
        "uptime_s": round(time.time() - start_time, 1) if start_time else 0,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "priority": PRIORITY})


def main():
    global MODEL, TOKENIZER, DEVICE, PRIORITY, start_time

    parser = argparse.ArgumentParser(
        description="HF Model HTTP Server with xsched priority (HIP/DTK)"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the HF model")
    parser.add_argument("--port", type=int, default=8080,
                        help="HTTP server port")
    parser.add_argument("--priority", type=int,
                        default=int(os.environ.get("XSCHED_AUTO_XQUEUE_PRIORITY", "0")),
                        help="xsched priority (also settable via XSCHED_AUTO_XQUEUE_PRIORITY)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID to use (default: 0, set to different IDs "
                             "for high/low priority servers)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    PRIORITY = args.priority

    # Check HIP device
    if not torch.cuda.is_available():
        print("[ERRO] No HIP/CUDA device available!")
        sys.exit(1)

    gpu_count = torch.cuda.device_count()
    if args.gpu >= gpu_count:
        print(f"[ERRO] GPU {args.gpu} not available (total: {gpu_count})")
        sys.exit(1)

    DEVICE = torch.device(f"cuda:{args.gpu}")
    props = torch.cuda.get_device_properties(DEVICE)
    print(f"[INFO] Device: {props.name} GPU#{args.gpu} ({props.total_memory // 1024 // 1024} MiB)")
    print(f"[INFO] xsched priority: {PRIORITY}")

    # Load model — pin to single GPU, no cross-device splitting
    print(f"[INFO] Loading tokenizer from {args.model}...")
    TOKENIZER = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    print(f"[INFO] Loading model from {args.model}...")
    MODEL = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map={"": DEVICE},
        trust_remote_code=True,
    )
    MODEL.eval()
    print(f"[INFO] Model loaded on {MODEL.device}")

    start_time = time.time()
    print(f"[INFO] Server starting on {args.host}:{args.port}")
    print(f"[INFO] Endpoints: /generate (POST), /stats (GET), /health (GET)")
    print(f"[INFO] ==============================================")
    print(f"[INFO] Ready. Send requests with:")
    print(f"  curl -X POST http://localhost:{args.port}/generate \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f'    -d \'{{"prompt": "你好", "max_new_tokens": 128}}\'')
    print(f"[INFO] ==============================================")

    # Use threaded=True so concurrent requests are handled properly
    app.run(host=args.host, port=args.port, threaded=True)


# Late import to avoid loading transformers on syntax check
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    main()
