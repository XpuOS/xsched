#!/usr/bin/env python3
"""
HuggingFace model benchmark for xsched priority scheduling test (HIP/DTK).

Adapted from benchmark_hf_xsched_with_lock.py for the Haiguang DTK platform.
Uses torch.cuda.Stream() — xsched intercepts hipStreamCreate via LD_PRELOAD,
so any non-default stream automatically gets an xqueue with the configured priority.

Usage:
    # Single run, priority via env var
    XSCHED_AUTO_XQUEUE_PRIORITY=10 python benchmark_hf_hip.py --mode xsched --model /path/to/Qwen3-8B

    # Recommended: use run_priority_test_hf.sh to launch two instances automatically
"""

import os
import sys
import time
import argparse
import threading
import queue
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList


# --- xsched HIP shim path (adjust to your installation) ---
SHIM_PATH = "/workspace/vllm_xsched/output/lib/libshimhip.so"


class TokenLimitLogitsProcessor(LogitsProcessor):
    """Suppress EOS before max_length, force EOS at/after max_length."""

    def __init__(self, max_length: int, eos_token_id: int):
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len < self.max_length:
            scores[:, self.eos_token_id] = -float("inf")
        else:
            scores[:, :] = -float("inf")
            scores[:, self.eos_token_id] = 0
        return scores


def ensure_environment_and_reexec(args: argparse.Namespace):
    """Check and set xsched environment variables, re-exec if needed."""
    if args.mode != "xsched":
        return

    env = os.environ.copy()
    needs_restart = False

    # HIP shim preload
    ld_preload = os.environ.get("LD_PRELOAD", "")
    if SHIM_PATH not in ld_preload:
        print(f"[WARN] LD_PRELOAD missing {SHIM_PATH}, setting...")
        env["LD_PRELOAD"] = f"{SHIM_PATH}:{ld_preload}".strip(":")
        needs_restart = True

    # Library paths
    shim_dir = os.path.dirname(SHIM_PATH)
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if shim_dir not in ld_library_path:
        print(f"[WARN] LD_LIBRARY_PATH missing {shim_dir}, setting...")
        env["LD_LIBRARY_PATH"] = f"{shim_dir}:{ld_library_path}".strip(":")
        needs_restart = True

    # xsched env vars
    target_vars = {
        "XSCHED_SCHEDULER": "GLB",
        "XSCHED_AUTO_XQUEUE": "ON",
        "XSCHED_AUTO_XQUEUE_PRIORITY": str(args.priority),
    }
    for key, value in target_vars.items():
        if os.environ.get(key) != value:
            print(f"[WARN] {key} not set to {value}, setting...")
            env[key] = value
            needs_restart = True

    if needs_restart:
        print("--- Re-executing with corrected environment... ---")
        os.execve(sys.executable, [sys.executable] + sys.argv, env)


def worker(
    task_q: queue.Queue,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_len: int,
    max_new_tokens: int,
    results_queue: queue.Queue,
    lock: threading.Lock,
    token_counter: list,
    token_lock: threading.Lock,
):
    """
    Worker thread: pulls tasks from queue, runs generation on a dedicated
    HIP stream (which xsched intercepts for priority scheduling).
    """
    # Ensure CUDA context is initialized in this thread
    torch.cuda.set_device(model.device)
    _ = torch.cuda.current_stream()

    # Create a dedicated non-default stream.
    # xsched intercepts hipStreamCreate → XStreamCreate → auto-creates xqueue.
    stream = torch.cuda.Stream()

    while not task_q.empty():
        try:
            _ = task_q.get_nowait()
        except queue.Empty:
            break

        try:
            # Random input_ids as prompt
            input_ids = torch.randint(
                1000, tokenizer.vocab_size - 5000, (1, prompt_len), device=model.device
            )

            logits_processor = LogitsProcessorList([
                TokenLimitLogitsProcessor(prompt_len + max_new_tokens, tokenizer.eos_token_id),
            ])

            with lock:
                with torch.cuda.stream(stream):
                    start_time = time.perf_counter()

                    attention_mask = torch.ones_like(input_ids)

                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        top_p=None,
                        pad_token_id=tokenizer.eos_token_id,
                        logits_processor=logits_processor,
                    )

            stream.synchronize()
            latency = time.perf_counter() - start_time

            output_tokens = output_ids.shape[1] - prompt_len
            results_queue.put((output_tokens, latency))

            with token_lock:
                token_counter[0] += output_tokens

            task_q.task_done()

        except Exception as e:
            print(f"[ERRO] Worker thread error: {e}")


def benchmark(args: argparse.Namespace):
    """Main benchmark routine."""
    print("=" * 50)
    print("  HF Model Benchmark — xsched HIP Priority Test")
    print("=" * 50)
    print(f"  Model:        {args.model}")
    print(f"  Mode:         {args.mode}")
    print(f"  Priority:     {args.priority}")
    print(f"  Threads:      {args.num_threads}")
    print(f"  Requests:     {args.num_requests}")
    print(f"  Prompt len:   {args.prompt_len}")
    print(f"  Max new tok:  {args.max_new_tokens}")
    print("=" * 50)

    # --- 1. Load model ---
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loading model (this may take a while for large models)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[INFO] Model loaded on {model.device}")

    # --- 2. Prepare task queue ---
    task_queue = queue.Queue()
    for _ in range(args.num_requests):
        task_queue.put(1)

    results_queue = queue.Queue()
    threads = []
    model_lock = threading.Lock()
    token_counter = [0]
    token_lock = threading.Lock()

    # --- 3. Launch workers ---
    print(f"[INFO] Starting {args.num_threads} worker threads...")
    for _ in range(args.num_threads):
        t = threading.Thread(
            target=worker,
            args=(
                task_queue, model, tokenizer,
                args.prompt_len, args.max_new_tokens,
                results_queue, model_lock,
                token_counter, token_lock,
            ),
        )
        threads.append(t)

    overall_start = time.perf_counter()
    for t in threads:
        t.start()

    # --- 4. Monitor progress ---
    last_check_time = time.time()
    last_token_count = 0
    total_requests = args.num_requests

    try:
        while results_queue.qsize() < total_requests:
            completed = results_queue.qsize()
            progress = completed / total_requests
            bar_len = 30
            filled = int(round(bar_len * progress))
            bar = "█" * filled + "-" * (bar_len - filled)

            # Machine-readable progress for external monitoring
            print(f"XSched_Progress:{args.priority}:{completed}/{total_requests}", flush=True)

            current_time = time.time()
            if current_time - last_check_time >= 2.0:
                with token_lock:
                    current_tokens = token_counter[0]
                delta_tokens = current_tokens - last_token_count
                delta_time = current_time - last_check_time
                rate = delta_tokens / delta_time if delta_time > 0 else 0
                print(f"XSched_Rate:{args.priority}:{rate:.2f}", flush=True)
                last_check_time = current_time
                last_token_count = current_tokens

            print(f"\r  Progress: |{bar}| {completed}/{total_requests} ({progress*100:.1f}%)",
                  end="", flush=True)

            if not any(t.is_alive() for t in threads):
                break
            time.sleep(0.5)
    finally:
        print()

    for t in threads:
        t.join()
    overall_end = time.perf_counter()

    # --- 5. Report ---
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    if not results:
        print("[ERRO] All requests failed!")
        return

    total_output_tokens = sum(r[0] for r in results)
    total_time = overall_end - overall_start
    latencies = [r[1] for r in results]

    print("\n--- Performance Report ---")
    print(f"  Elapsed:          {total_time:.2f} s")
    print(f"  Completed:        {len(results)} requests")
    print(f"  Total out tokens: {total_output_tokens}")
    print(f"  Throughput:       {total_output_tokens / total_time:.2f} tokens/s")
    print(f"  Avg latency:      {np.mean(latencies) * 1000:.2f} ms")
    print(f"  P90 latency:      {np.percentile(latencies, 90) * 1000:.2f} ms")
    print(f"  P99 latency:      {np.percentile(latencies, 99) * 1000:.2f} ms")

    p99_ms = np.percentile(latencies, 99) * 1000
    print(f"XSched_Result:{args.priority}:{len(results)}:{total_time:.2f}:{p99_ms:.2f}", flush=True)
    print("--- Done ---")


def main():
    parser = argparse.ArgumentParser(
        description="HF model benchmark with xsched priority scheduling (HIP/DTK)"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model (e.g., /path/to/Qwen3-8B)")
    parser.add_argument("--mode", type=str, choices=["direct", "xsched"], required=True,
                        help="'direct' for native, 'xsched' for xsched scheduling")
    parser.add_argument("--num-threads", type=int, default=4,
                        help="Number of concurrent worker threads")
    parser.add_argument("--num-requests", type=int, default=32,
                        help="Total number of inference requests")
    parser.add_argument("--prompt-len", type=int, default=512,
                        help="Prompt length in tokens")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Max tokens to generate per request")
    parser.add_argument("--priority", type=int, default=0,
                        help="[xsched mode] Priority value (-256 to 255, higher = higher priority)")

    args = parser.parse_args()

    ensure_environment_and_reexec(args)
    benchmark(args)


if __name__ == "__main__":
    main()
