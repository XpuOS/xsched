"""
KV cache 抢占测试。

原理:
  1. 先发大量低优先级 (pri=10) 请求，让它们占满 KV cache。
  2. 再发高优先级 (pri=0) 请求，此时 KV cache 已满，
     scheduler 会抢占 (preempt) 低优请求，释放 block 给高优请求。
  3. 被抢占的请求 num_computed_tokens=0，需要从头重算，延迟显著增加。

服务端要求:
  vllm serve <model> --scheduling-policy priority --gpu-memory-utilization 0.3

用法:
  python vllm_preempt_test.py --model Qwen3-8B --port 8001 --num 30 --concurrency 16
"""
import argparse
import sys
import time
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# 长 prompt (~400 token)，每个请求消耗约 25 个 KV cache block (block_size=16)
LONG_PROMPT = (
    "History: " + "verify " * 400 + ". "
    "Question: Explain quantum physics in detail, covering "
    "wave-particle duality, the uncertainty principle, quantum "
    "entanglement, and the measurement problem."
)
GEN_LEN = 256  # 每个请求生成的 token 数


def send_one(url, model, priority, req_id):
    """发送一个请求，返回 (req_id, priority, latency_ms, num_preemptions)。"""
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": LONG_PROMPT},
        ],
        "max_tokens": GEN_LEN,
        "temperature": 0.0,
        "priority": priority,
    }

    start_t = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        end_t = time.perf_counter()
        latency_ms = (end_t - start_t) * 1000
        body = resp.json()
        num_preempt = body.get("usage", {}).get("num_preemptions", 0)
        return (req_id, priority, latency_ms, num_preempt, None)
    except Exception as e:
        return (req_id, priority, -1, -1, str(e)[:200])


def main():
    parser = argparse.ArgumentParser(description="vLLM KV cache preemption test")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--num", type=int, default=30,
                        help="每个优先级发送的请求数")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Phase 1 的并发度")
    parser.add_argument("--phase1-delay", type=float, default=1.0,
                        help="Phase 1 发送完毕后，等待多少秒再发 Phase 2")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}/v1/chat/completions"

    print(f"{'='*60}")
    print("KV CACHE PREEMPTION TEST")
    print(f"  Model: {args.model}  |  Port: {args.port}")
    print(f"  Phase 1: {args.num} x LOW(pri=10)  — 先占满 KV cache")
    print(f"  Wait:    {args.phase1_delay}s")
    print(f"  Phase 2: {args.num} x HIGH(pri=0) — 触发抢占")
    print(f"  Prompt:  ~400 tokens  |  Output: {GEN_LEN} tokens")
    print(f"{'='*60}")

    all_results = []  # (req_id, priority, latency_ms, num_preempt, error)

    with ThreadPoolExecutor(max_workers=args.concurrency * 2) as executor:

        # === Phase 1: 发低优先级请求 ===
        print("\n── Phase 1: 发送 LOW(pri=10) 请求 ──")
        low_futures = {
            executor.submit(send_one, url, args.model, 10, i): i
            for i in range(args.num)
        }
        # 等一小段时间，确保低优请求进入 running 并占住 KV cache
        print(f"  (等待 {args.phase1_delay}s，让低优请求分配 KV cache block...)")
        time.sleep(args.phase1_delay)

        # === Phase 2: 发高优先级请求 ===
        print("── Phase 2: 发送 HIGH(pri=0) 请求 ──")
        high_futures = {
            executor.submit(send_one, url, args.model, 0, i): i
            for i in range(args.num, args.num * 2)
        }

        # 收集结果
        for fut in as_completed(low_futures | high_futures):
            req_id, prio, lat, preempt, err = fut.result()
            all_results.append((req_id, prio, lat, preempt, err))
            if err:
                print(f"  [{'HIGH' if prio == 0 else 'LOW'}] req={req_id:3d} FAILED: {err}")
            else:
                marker = f" ⚡ PREEMPTED x{preempt}" if preempt > 0 else ""
                print(f"  [{'HIGH' if prio == 0 else 'LOW'}] req={req_id:3d} | "
                      f"lat={lat:8.0f}ms{marker}")

    # === 统计 ===
    high = [(lat, preempt) for _, prio, lat, preempt, err in all_results
            if prio == 0 and err is None]
    low = [(lat, preempt) for _, prio, lat, preempt, err in all_results
           if prio == 10 and err is None]

    if not high or not low:
        print("❌ 结果不足，无法对比")
        sys.exit(1)

    hl = np.array([x[0] for x in high])
    ll = np.array([x[0] for x in low])
    hp = sum(1 for x in high if x[1] > 0)
    lp = sum(1 for x in low if x[1] > 0)
    hp_total = sum(x[1] for x in high)
    lp_total = sum(x[1] for x in low)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'HIGH(pri=0)':>15} {'LOW(pri=10)':>15}")
    print(f"{'─'*25} {'─'*15} {'─'*15}")
    for name, fn in [
        ("Avg latency (ms)", np.mean),
        ("P50 latency (ms)", lambda x: np.percentile(x, 50)),
        ("P90 latency (ms)", lambda x: np.percentile(x, 90)),
        ("P99 latency (ms)", lambda x: np.percentile(x, 99)),
    ]:
        print(f"{name:<25} {fn(hl):15.1f} {fn(ll):15.1f}")
    print(f"{'Preempted req count':<25} {hp:>15} {lp:>15}")
    print(f"{'Total preemptions':<25} {hp_total:>15} {lp_total:>15}")
    print(f"{'─'*60}")

    if lp > hp or lp_total > hp_total:
        print(f"✓ KV cache 抢占生效: LOW 被抢占 {lp} 个请求 / {lp_total} 次 "
              f"vs HIGH {hp} 个请求 / {hp_total} 次")
        print(f"  被抢占的请求需要从头重算 KV cache → 延迟更高")
    else:
        print("✗ 未触发抢占。请检查:")
        print("  1. vllm serve ... --scheduling-policy priority")
        print("  2. 试用 --gpu-memory-utilization 0.3 限制 KV cache 大小")
        print("  3. 增大 --num 或 --concurrency")


if __name__ == "__main__":
    main()
