import argparse
import time
import requests
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# === 配置 ===
# 使用与 dual_client 完全一致的配置
LONG_USER_INPUT = "History: " + "verify " * 400 + ". Question: Explain quantum physics?"
GEN_LEN = 256 

class BenchmarkClient:
    def __init__(self, port, model, label, priority=None):
        self.base_url = f"http://localhost:{port}/v1/chat/completions"
        self.model = model
        self.label = label
        self.priority = priority
        self.latencies = []
        self.throughputs = []
        self.errors = 0

    def send_request(self, req_id):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": LONG_USER_INPUT}
            ],
            "max_tokens": GEN_LEN,
            "temperature": 0.0,
            "ignore_eos": True
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
            print(f"   [{self.label}] Req {req_id:3d} Done | Latency: {latency_ms:.2f} ms | Throughput: {tps:.2f} tokens/s")
            
        except Exception as e:
            self.errors += 1
            print(f"   ❌ Req {req_id} Failed: {str(e)[:100]}")

    def run_benchmark(self, num_requests, concurrency):
        print(f"\n🚀 Starting Single-vLLM Benchmark")
        print(f"   Target: {self.base_url} | Model: {self.model}")
        print(f"   Load:   {num_requests} total requests, {concurrency} concurrent")
        print("=" * 60)

        start_bench = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # 引入极小的延迟打散请求，防止瞬间并发冲垮 vLLM 调度器
            futures = []
            for i in range(num_requests):
                time.sleep(0.1) 
                futures.append(executor.submit(self.send_request, i))
                
            for f in futures:
                f.result()
        end_bench = time.time()
        
        duration = end_bench - start_bench
        self.print_report(duration)

    def print_report(self, duration):
        if not self.latencies:
            print("❌ No successful requests.")
            return

        lats = np.array(self.latencies)
        tps_list = np.array(self.throughputs)
        print("\n" + "="*60)
        print(f"📊 vLLM Performance Report")
        print("="*60)
        print(f"Total Duration:  {duration:.2f} s")
        print(f"System Throughput: {len(lats)/duration:.2f} RPS")
        print(f"Avg Latency:     {np.mean(lats):.2f} ms")
        print(f"Avg Throughput:  {np.mean(tps_list):.2f} tokens/s")
        print(f"P50 Latency:     {np.percentile(lats, 50):.2f} ms")
        print(f"P99 Latency:     {np.percentile(lats, 99):.2f} ms")
        print(f"Errors:          {self.errors}")
        print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--port", type=int, default=8001)
    # 降低默认请求数，对齐 dual_client 的单边压力
    parser.add_argument("--num", type=int, default=10) 
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--priority", type=int, default=None)
    args = parser.parse_args()

    client = BenchmarkClient(args.port, args.model, "vLLM-Service", args.priority)
    client.run_benchmark(args.num, args.concurrency)
