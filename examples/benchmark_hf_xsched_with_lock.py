import os
import sys
import time
import argparse
import random
import threading
import queue
import ctypes
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

# --- 配置区 ---
SHIM_PATH = "/root/workspace/xsched/output/lib/libshimcuda.so"

# --- 工具与辅助类 ---

class TokenLimitLogitsProcessor(LogitsProcessor):
    """
    一个 LogitsProcessor，在达到最大长度之前抑制 EOS token，在达到最大长度时强制生成 EOS token。
    """
    def __init__(self, max_length: int, eos_token_id: int):
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        # 如果当前长度小于最大长度，则抑制 EOS token
        if cur_len < self.max_length:
            scores[:, self.eos_token_id] = -float("inf")
        # 如果当前长度等于或超过最大长度，则强制 EOS token
        else:
            scores[:, :] = -float("inf")
            scores[:, self.eos_token_id] = 0
        return scores

def ensure_environment_and_reexec(args: argparse.Namespace):
    """
    在 xsched 模式下，检查并设置必要的环境变量，如果需要则重启脚本。
    """
    if args.mode != 'xsched':
        return

    ld_preload = os.environ.get('LD_PRELOAD', '')
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    shim_dir = os.path.dirname(SHIM_PATH)

    env = os.environ.copy()
    needs_restart = False

    if SHIM_PATH not in ld_preload:
        print(f"⚠️ 警告: LD_PRELOAD 未加载 {SHIM_PATH}。正在设置...")
        env["LD_PRELOAD"] = f"{SHIM_PATH}:{ld_preload}".strip(':')
        needs_restart = True

    if shim_dir not in ld_library_path:
        print(f"⚠️ 警告: LD_LIBRARY_PATH 未包含 {shim_dir}。正在设置...")
        env["LD_LIBRARY_PATH"] = f"{shim_dir}:{ld_library_path}".strip(':')
        needs_restart = True

    # 检查 XSCHED_* 变量，确保它们在重启时被设置
    target_xsched_vars = {
        "XSCHED_SCHEDULER": "GLB",
        "XSCHED_AUTO_XQUEUE": "ON",
        "XSCHED_AUTO_XQUEUE_PRIORITY": str(args.priority)
    }
    for key, value in target_xsched_vars.items():
        if os.environ.get(key) != value:
            print(f"⚠️ 警告: {key} 未设置为 {value}。正在设置...")
            env[key] = value
            needs_restart = True

    if needs_restart:
        print("--- 正在使用正确的环境重新执行脚本... ---")
        os.execve(sys.executable, [sys.executable] + sys.argv, env)

def create_xsched_stream() -> torch.cuda.Stream:
    """通过 ctypes 直接调用 shim 库来创建被 xsched 拦截的流。"""
    if not os.path.exists(SHIM_PATH):
        print(f"❌ 错误: 找不到 Shim 库: {SHIM_PATH}")
        return None
    try:
        libshim = ctypes.CDLL(SHIM_PATH)
        stream_ptr = ctypes.c_void_p()
        # 调用 cuStreamCreate (Flags=0, 非阻塞)
        ret = libshim.cuStreamCreate(ctypes.byref(stream_ptr), 0)
        if ret != 0:
            print(f"❌ Shim 创建流失败，错误码: {ret}")
            return None
        # print(f"✅ XSched 流创建成功！Handle: {hex(stream_ptr.value)}")
        return torch.cuda.ExternalStream(stream_ptr.value)
    except Exception as e:
        print(f"❌ 创建 XSched 流时发生异常: {e}")
        return None

def worker(
    q: queue.Queue,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_len: int,
    max_new_tokens: int,
    use_xsched_stream: bool,
    results_queue: queue.Queue,
    lock: threading.Lock,
    token_counter: list,
    token_lock: threading.Lock
):
    """工作线程，从队列中获取任务并执行推理。"""
    # [关键修复] 在每个工作线程中，必须先初始化该线程的 CUDA 上下文，
    # 然后才能调用任何 CUDA Driver API (如 create_xsched_stream 中的 cuStreamCreate)。
    # PyTorch 会在第一次在该线程上执行 CUDA 操作时懒惰地创建上下文。
    if use_xsched_stream:
        # model.device 是主线程中模型所在的设备
        torch.cuda.set_device(model.device)
        # 执行一个无害的 CUDA 操作来触发上下文初始化
        _ = torch.cuda.current_stream()

    # 为每个线程创建独立的流
    stream = create_xsched_stream() if use_xsched_stream else torch.cuda.current_stream()

    # [健壮性] 如果流创建失败，则无法继续执行
    if stream is None and use_xsched_stream:
        print(f"❌ 线程 {threading.get_ident()}: XSched 流创建失败，无法继续。")
        return

    while not q.empty():
        try:
            _ = q.get_nowait()
            
            # 1. 生成随机 prompt
            input_ids = torch.randint(1000, tokenizer.vocab_size - 5000, (1, prompt_len), device=model.device)
            
            # 2. 设置 logits processor 以便精确控制长度
            logits_processor = LogitsProcessorList([
                TokenLimitLogitsProcessor(prompt_len + max_new_tokens, tokenizer.eos_token_id),
            ])

            # 3. 在锁的保护下，在指定的流上执行生成
            with lock:

                with torch.cuda.stream(stream):
                    
                    start_time = time.perf_counter()

                    # 创建 attention_mask，因为我们的 prompt 没有 padding，所以全为 1
                    attention_mask = torch.ones_like(input_ids)

                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,  # 显式传递 attention_mask
                        max_new_tokens=max_new_tokens,
                        do_sample=False, # 关闭采样以获得可复现的性能
                        top_p=None,      # 明确禁用 top_p 以消除警告
                        pad_token_id=tokenizer.eos_token_id, # 显式设置 pad_token_id
                        logits_processor=logits_processor
                    )
            
            # 同步当前流以确保任务完成并获得准确的延迟
            stream.synchronize()
            latency = time.perf_counter() - start_time

            output_tokens = output_ids.shape[1] - prompt_len
            results_queue.put((output_tokens, latency))
            
            with token_lock:
                token_counter[0] += output_tokens
            
            q.task_done()
        except queue.Empty:
            break
        except Exception as e:
            print(f"线程出错: {e}")
            break

def benchmark(args: argparse.Namespace):
    """主压测逻辑。"""
    print("--- HuggingFace 原生模型压测开始 ---")
    print(f"模型: {args.model}")
    print(f"模式: {args.mode}")
    print(f"并发线程数: {args.num_threads}")
    print(f"总请求数: {args.num_requests}")
    print(f"Prompt 长度: {args.prompt_len} tokens")
    print(f"生成长度: {args.max_new_tokens} tokens")

    # --- 1. 初始化模型和分词器 ---
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # --- 2. 准备任务队列 ---
    task_queue = queue.Queue()
    for _ in range(args.num_requests):
        task_queue.put(1)

    results_queue = queue.Queue()
    threads = []
    model_lock = threading.Lock()
    
    # Global token counter for rate tracking
    token_counter = [0]
    token_lock = threading.Lock()
    
    use_xsched = (args.mode == 'xsched')

    print("正在启动工作线程...")
    for _ in range(args.num_threads):
        thread = threading.Thread(
            target=worker,
            args=(task_queue, model, tokenizer, args.prompt_len, args.max_new_tokens, use_xsched, results_queue, model_lock, token_counter, token_lock)
        )
        threads.append(thread)

    # --- 3. 开始压测 ---
    print("压测开始！")
    overall_start_time = time.perf_counter() 

    for t in threads:
        t.start()

    # --- 3.5 监控进度 ---
    num_total_requests = args.num_requests
    last_check_time = time.time()
    last_token_count = 0

    try:
        while results_queue.qsize() < num_total_requests:
            completed_count = results_queue.qsize()
            progress = completed_count / num_total_requests
            bar_len = 30
            filled_len = int(round(bar_len * progress))
            bar = '█' * filled_len + '-' * (bar_len - filled_len)

            # Machine readable output for server parsing
            print(f"XSched_Progress:{args.priority}:{completed_count}/{num_total_requests}", flush=True)

            # Rate Calculation
            current_time = time.time()
            if current_time - last_check_time >= 2.0: # Check every 2 seconds
                 with token_lock:
                     current_tokens = token_counter[0]
                 
                 delta_tokens = current_tokens - last_token_count
                 delta_time = current_time - last_check_time
                 rate = delta_tokens / delta_time if delta_time > 0 else 0
                 
                 print(f"XSched_Rate:{args.priority}:{rate:.2f}", flush=True)
                 
                 last_check_time = current_time
                 last_token_count = current_tokens

            # print(f"\r进度: |{bar}| {completed_count}/{num_total_requests} ({progress*100:.1f}%)", end="", flush=True)

            # 如果所有线程都意外结束了，就退出循环
            if not any(t.is_alive() for t in threads):
                break
            time.sleep(0.5) # 每 0.5 秒刷新一次
    finally:
        print() # 确保光标移动到新的一行

    # 等待所有线程最终结束（通常此时已结束）
    for t in threads:
        t.join()
    overall_end_time = time.perf_counter()

    # --- 4. 收集并报告性能 ---
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    total_output_tokens = sum(r[0] for r in results)
    total_time = overall_end_time - overall_start_time
    latencies = [r[1] for r in results]

    if not results:
        print("所有请求都失败了！")
        return

    print("\n--- 性能报告 ---")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"完成的请求数: {len(results)}")
    print(f"总输出 Tokens: {total_output_tokens}")
    print("-" * 20)
    print(f"吞吐率 (Throughput): {total_output_tokens / total_time:.2f} tokens/s")
    print("-" * 20)
    print(f"平均延迟 (Avg. Latency): {np.mean(latencies) * 1000:.2f} ms")
    print(f"P90 延迟: {np.percentile(latencies, 90) * 1000:.2f} ms")
    print(f"P99 延迟: {np.percentile(latencies, 99) * 1000:.2f} ms")
    
    # Machine readable result for server
    p99_ms = np.percentile(latencies, 99) * 1000
    print(f"XSched_Result:{args.priority}:{len(results)}:{total_time:.2f}:{p99_ms:.2f}", flush=True)

    print("--- 压测结束 ---")

def main_cli():
    parser = argparse.ArgumentParser(description="HuggingFace 原生模型性能压测脚本 (支持 XSched)")
    parser.add_argument("--model", type=str, default="/root/models/Qwen1.5-0.5B-Chat", help="模型路径")
    parser.add_argument("--mode", type=str, choices=['direct', 'xsched'], required=True, help="运行模式: 'direct' (原生) 或 'xsched' (通过XSched调度)")
    parser.add_argument("--num-threads", type=int, default=16, help="并发工作线程数量")
    parser.add_argument("--num-requests", type=int, default=64, help="总请求数量")
    parser.add_argument("--prompt-len", type=int, default=2000, help="输入 Prompt 的 Token 长度")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="期望生成的 Token 长度")
    parser.add_argument("--priority", type=int, default=0, help="[仅XSched模式] 任务优先级")
    
    args = parser.parse_args()
    
    ensure_environment_and_reexec(args)

    benchmark(args)

if __name__ == "__main__":
    main_cli()