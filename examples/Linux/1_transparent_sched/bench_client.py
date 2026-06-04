#!/usr/bin/env python3
"""
Simple benchmark client for model_server.py.

Sends concurrent requests to a model server endpoint and reports throughput/latency.

Usage:
    python bench_client.py --url http://localhost:8080/generate --concurrency 4 --requests 32

The server prints per-request latency in its own terminal.
This client prints aggregate throughput/latency summary.
"""

import time
import json
import argparse
import threading
import queue
import urllib.request
import urllib.error
import numpy as np


def send_request(url: str, prompt: str, max_new_tokens: int, timeout: int) -> dict:
    """Send a single request and return timing info."""
    data = json.dumps({
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = json.loads(e.read().decode("utf-8"))
        raise RuntimeError(f"HTTP {e.code}: {body.get('error', 'unknown')}")
    latency = time.perf_counter() - t0

    return {
        "latency_ms": latency * 1000,
        "output_tokens": body.get("output_tokens", 0),
        "tokens_per_sec": body.get("output_tokens", 0) / latency if latency > 0 else 0,
    }


def worker(url: str, prompt: str, max_new_tokens: int, timeout: int,
           task_q: queue.Queue, result_q: queue.Queue):
    """Worker thread: sends requests and collects results."""
    while not task_q.empty():
        try:
            _ = task_q.get_nowait()
        except queue.Empty:
            break
        try:
            r = send_request(url, prompt, max_new_tokens, timeout)
            result_q.put(r)
        except Exception as e:
            print(f"[ERRO] Request failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Model server benchmark client")
    parser.add_argument("--url", type=str, required=True,
                        help="Server endpoint URL (e.g., http://localhost:8080/generate)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of concurrent request threads")
    parser.add_argument("--requests", type=int, default=32,
                        help="Total number of requests to send")
    parser.add_argument("--prompt", type=str, default="请用中文简要介绍深度学习的基本原理。",
                        help="Prompt text to send")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Max tokens to generate per request")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Request timeout in seconds")
    args = parser.parse_args()

    print(f"[INFO] Target: {args.url}")
    print(f"[INFO] Concurrency: {args.concurrency}, Requests: {args.requests}")
    print(f"[INFO] Max new tokens: {args.max_new_tokens}")

    task_q = queue.Queue()
    for _ in range(args.requests):
        task_q.put(1)

    result_q = queue.Queue()

    print(f"[INFO] Sending {args.requests} requests with {args.concurrency} threads...")
    t0 = time.perf_counter()

    threads = []
    for _ in range(args.concurrency):
        t = threading.Thread(
            target=worker,
            args=(args.url, args.prompt, args.max_new_tokens, args.timeout, task_q, result_q),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    total_time = time.perf_counter() - t0

    results = []
    while not result_q.empty():
        results.append(result_q.get())

    if not results:
        print("[ERRO] All requests failed!")
        return

    latencies = [r["latency_ms"] for r in results]
    total_tokens = sum(r["output_tokens"] for r in results)

    print(f"\n--- Results ({args.url}) ---")
    print(f"  Completed:      {len(results)}/{args.requests} requests")
    print(f"  Total time:     {total_time:.1f} s")
    print(f"  Throughput:     {len(results) / total_time:.2f} req/s")
    print(f"  Token rate:     {total_tokens / total_time:.2f} tokens/s")
    print(f"  Total tokens:   {total_tokens}")
    print(f"  Avg latency:    {np.mean(latencies):.0f} ms")
    print(f"  P50 latency:    {np.percentile(latencies, 50):.0f} ms")
    print(f"  P90 latency:    {np.percentile(latencies, 90):.0f} ms")
    print(f"  P99 latency:    {np.percentile(latencies, 99):.0f} ms")
    print(f"  Min/Max:        {np.min(latencies):.0f} / {np.max(latencies):.0f} ms")


if __name__ == "__main__":
    main()
