import argparse
import time
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# === 配置 ===
# 长 prompt: 约 400 token，消耗约 25 个 KV cache block (block_size=16)
# 用于触发 KV cache 不足 → 抢占
LONG_PROMPT = "History: " + "verify " * 400 + ". Question: Explain quantum physics?"
# 短 prompt: 约 20 token，KV cache 压力小
SHORT_PROMPT = "Question: What is 2+2? Answer briefly."
GEN_LEN = 1024


class BenchmarkClient:
    def __init__(self, port, model, label, priority=None, use_long_prompt=True):
        self.base_url = f"http://localhost:{port}/v1/chat/completions"
        self.model = model
        self.label = label
        self.priority = priority
        self.prompt = LONG_PROMPT if use_long_prompt else SHORT_PROMPT
        self.latencies = []
        self.throughputs = []
        self.preemptions = []  # 每个请求被抢占的次数
        self.errors = 0

    def send_request(self, req_id):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": self.prompt},
            ],
            "max_tokens": GEN_LEN,
            "temperature": 0.0,
            "ignore_eos": True,
        }
        if self.priority is not None:
            payload["priority"] = self.priority

        start_t = time.perf_counter()
        try:
            resp = requests.post(self.base_url, json=payload, timeout=300)
            resp.raise_for_status()
            end_t = time.perf_counter()

            duration_s = end_t - start_t
            latency_ms = duration_s * 1000
            tps = GEN_LEN / duration_s if duration_s > 0 else 0

            self.latencies.append(latency_ms)
            self.throughputs.append(tps)

            # 尝试解析 vLLM 返回的 preemption 统计
            body = resp.json()
            num_preempt = body.get("usage", {}).get("num_preemptions", 0)
            self.preemptions.append(num_preempt)

            preempt_str = f"preempt={num_preempt}" if num_preempt > 0 else ""
            print(
                f"   [{self.label}] Req {req_id:3d} Done | "
                f"Latency: {latency_ms:8.2f} ms | "
                f"Throughput: {tps:.2f} tok/s  {preempt_str}"
            )

        except Exception as e:
            self.errors += 1
            print(f"   ❌ [{self.label}] Req {req_id} Failed: {str(e)[:100]}")

    def run_benchmark(self, num_requests, concurrency, arrival_gap_s=0.0):
        prompt_type = "LONG (~400 tok)" if self.prompt == LONG_PROMPT else "SHORT (~20 tok)"
        print(f"\n🚀 Starting Benchmark [{self.label}]")
        print(f"   Target: {self.base_url} | Model: {self.model}")
        print(f"   Priority: {self.priority} | Prompt: {prompt_type}")
        print(f"   Load:   {num_requests} requests, {concurrency} concurrent")
        print("=" * 60)

        start_bench = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for i in range(num_requests):
                if arrival_gap_s > 0:
                    time.sleep(arrival_gap_s)
                futures.append(executor.submit(self.send_request, i))

            for f in futures:
                f.result()
        end_bench = time.time()

        duration = end_bench - start_bench
        self.print_report(duration)

    def print_report(self, duration):
        if not self.latencies:
            print(f"   [{self.label}] ❌ No successful requests.")
            return

        lats = np.array(self.latencies)
        tps_list = np.array(self.throughputs)
        preempted = sum(1 for p in self.preemptions if p > 0)
        print("\n" + "=" * 60)
        print(f"📊 Report [{self.label}]  (priority={self.priority})")
        print("=" * 60)
        print(f"  Total Duration:    {duration:.2f} s")
        print(f"  Throughput:        {len(lats)/duration:.2f} RPS")
        print(f"  Avg Latency:       {np.mean(lats):8.2f} ms")
        print(f"  P50 Latency:       {np.percentile(lats, 50):8.2f} ms")
        print(f"  P90 Latency:       {np.percentile(lats, 90):8.2f} ms")
        print(f"  P99 Latency:       {np.percentile(lats, 99):8.2f} ms")
        print(f"  Min/Max Latency:   {np.min(lats):.0f} / {np.max(lats):.0f} ms")
        print(f"  Preempted reqs:    {preempted}/{len(self.preemptions)}")
        print(f"  Errors:            {self.errors}")
        print("-" * 60)


def run_combined_test(port, model, num, concurrency, use_long):
    """同时跑 priority=0 和 priority=10，直接对比延迟。"""
    print(f"\n{'='*60}")
    print("COMBINED PRIORITY TEST")
    print(f"  HIGH (pri=0)  vs  LOW (pri=10)  — 同时发起")
    print(f"  Each: {num} requests, {concurrency} concurrent")
    print(f"{'='*60}")

    high_client = BenchmarkClient(port, model, "HIGH(pri=0)", priority=0,
                                  use_long_prompt=use_long)
    low_client = BenchmarkClient(port, model, "LOW(pri=10)", priority=10,
                                 use_long_prompt=use_long)

    start_bench = time.time()
    with ThreadPoolExecutor(max_workers=concurrency * 2) as executor:
        future_map = {}
        idx = 0
        for _ in range(num):
            f_high = executor.submit(high_client.send_request, idx)
            future_map[f_high] = ("HIGH", idx)
            idx += 1
            f_low = executor.submit(low_client.send_request, idx)
            future_map[f_low] = ("LOW", idx)
            idx += 1

        for fut in as_completed(future_map):
            label, req_id = future_map[fut]
            fut.result()  # 结果已在 send_request 中记录

    duration = time.time() - start_bench
    high_client.print_report(duration)
    low_client.print_report(duration)

    # === 对比 ===
    if high_client.latencies and low_client.latencies:
        hl = np.array(high_client.latencies)
        ll = np.array(low_client.latencies)
        hp = sum(1 for p in high_client.preemptions if p > 0)
        lp = sum(1 for p in low_client.preemptions if p > 0)

        print(f"\n{'='*60}")
        print("HEAD-TO-HEAD COMPARISON")
        print(f"{'='*60}")
        print(f"{'Metric':<25} {'HIGH(pri=0)':>15} {'LOW(pri=10)':>15} {'Ratio':>10}")
        print(f"{'─'*25} {'─'*15} {'─'*15} {'─'*10}")
        for name, fn in [
            ("Avg latency (ms)", np.mean),
            ("P50 latency (ms)", lambda x: np.percentile(x, 50)),
            ("P90 latency (ms)", lambda x: np.percentile(x, 90)),
            ("P99 latency (ms)", lambda x: np.percentile(x, 99)),
            ("Max latency (ms)", np.max),
        ]:
            hv, lv = fn(hl), fn(ll)
            ratio = lv / hv if hv > 0 else float("inf")
            print(f"{name:<25} {hv:15.1f} {lv:15.1f} {ratio:9.2f}x")
        print(f"{'Preempted count':<25} {hp:>15} {lp:>15}")
        print(f"{'─'*60}")

        if ll.mean() > hl.mean() * 1.3:
            print(f"✓ 优先级调度生效: LOW 延迟是 HIGH 的 {ll.mean()/hl.mean():.1f} 倍")
        elif lp > hp:
            print(f"✓ 优先级调度通过抢占生效: LOW 被抢占 {lp} 次 vs HIGH {hp} 次")
        else:
            print("✗ 无明显差异。请检查:")
            print("  1. vLLM 是否用 --scheduler-policy priority 启动?")
            print("  2. --max-num-seqs 是否足够小 (建议 4)?")
            print("  3. KV cache 是否被耗尽? (用 --long-prompt + 增大 --num)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--num", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--priority", type=int, default=None,
                        help="单优先级模式: 所有请求使用此 priority")
    parser.add_argument("--combined", action="store_true",
                        help="同时跑 priority=0 和 priority=10，直接对比延迟")
    parser.add_argument("--short-prompt", action="store_true",
                        help="用短 prompt (~20 tok)，减少 KV cache 压力")
    args = parser.parse_args()

    use_long = not args.short_prompt

    if args.combined:
        run_combined_test(args.port, args.model, args.num, args.concurrency, use_long)
    else:
        prio = args.priority if args.priority is not None else 0
        client = BenchmarkClient(args.port, args.model, "vLLM-Service", prio, use_long)
        client.run_benchmark(args.num, args.concurrency)
