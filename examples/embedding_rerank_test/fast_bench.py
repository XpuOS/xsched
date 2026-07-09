import sys
import time
import requests
import threading
import argparse
import base64
import os
from concurrent.futures import ThreadPoolExecutor

# 背景负载使用固定的长文本
LONG_TEXT = "offline background load text content " * 20

def get_image_base64(image_path):
    if not image_path or not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def background_worker(task, port, duration, worker_id, image_b64=None):
    """
    worker_id: 用于区分同一任务下的不同并发线程
    """
    url = f"http://127.0.0.1:{port}/v1/predict"
    
    # 遵循 Qwen3VLRequest 的格式
    if task == "embedding":
        payload = {
            "task": task,
            "inputs": [{"text": LONG_TEXT}]
        }
    elif task == "rerank":
        payload = {
            "task": task,
            "query": "What is the capital of France?",
            "documents": [LONG_TEXT, "Short text content"]
        }
    else:
        payload = {"task": task, "text": LONG_TEXT}

    if image_b64 and task in ["clip", "ocr"]:
        payload["image_base64"] = image_b64
    
    end_time = time.time() + duration
    count = 0
    total_lat = 0
    
    while time.time() < end_time:
        try:
            # 记录请求开始时间
            start_t = time.perf_counter()
            r = requests.post(url, json=payload, timeout=20)
            
            if r.status_code == 200:
                latency_ms = (time.perf_counter() - start_t) * 1000
                count += 1
                total_lat += latency_ms

                print(f"   [{task.upper()}] Worker {worker_id:2d} | Req {count:4d} Done | Latency: {latency_ms:.2f} ms")
            else:
                print(f"   ⚠️ [{task.upper()}] Worker {worker_id:2d} Error {r.status_code}: {r.text[:200]}")
                time.sleep(1) 
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print(f"   ⚠️ [{task.upper()}] Worker {worker_id:2d} Exception: {e}")
            
    return count, total_lat

def main():
    parser = argparse.ArgumentParser(description="Offline Background Load Generator")
    parser.add_argument("duration", type=int, help="Duration in seconds")
    parser.add_argument("concurrency", type=int, help="Concurrency per task")
    parser.add_argument("--tasks", type=str, default="embedding,clip,ocr,rerank", 
                        help="Comma-separated tasks to run (e.g., embedding,clip)")
    parser.add_argument("--image", type=str, default="docs/img/xsched-logo.png")
    args = parser.parse_args()
    
    image_b64 = get_image_base64(args.image)
    task_map = {
        "embedding": 8891,
        "rerank": 8892,
        "clip": 8893,
        "ocr": 8894
    }
    
    active_tasks = [t.strip() for t in args.tasks.split(",") if t.strip() in task_map]
    
    print(f"🔥 Generating OFFLINE load for: {active_tasks}")
    print(f"🔥 Duration: {args.duration}s, Concurrency per task: {args.concurrency}")
    print("=" * 65)

    results = {}
    # 计算总线程数
    with ThreadPoolExecutor(max_workers=len(active_tasks) * args.concurrency) as executor:
        futures_map = {}
        for task in active_tasks:
            port = task_map[task]
            futures = []
            for i in range(args.concurrency):
                # 传入 i 作为 worker_id 方便观察并发情况
                futures.append(executor.submit(background_worker, task, port, args.duration, i, image_b64))
            futures_map[task] = futures
        
        for task, futures in futures_map.items():
            task_total_req = 0
            task_total_lat = 0
            for f in futures:
                count, lat = f.result()
                task_total_req += count
                task_total_lat += lat
            results[task] = (task_total_req, task_total_lat)

    # 最终报告
    print("\n🏆 FINAL RESULTS (Background Load):")
    print("-" * 65)
    print(f"{'Task':<12}\t{'Requests':<8}\t{'Throughput(RPS)':<15}\t{'Avg Latency'}")
    for task, (req, lat) in results.items():
        rps = req / args.duration
        avg = lat / req if req > 0 else 0
        print(f"{task:<12}\t{req:<8}\t{rps:<15.2f}\t{avg:<10.2f} ms")
    print("-" * 65)

if __name__ == "__main__":
    main()