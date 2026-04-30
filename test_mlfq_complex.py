import torch
import time
import os
import sys
import random

custom_stream = torch.cuda.Stream()
task_type = sys.argv[1] if len(sys.argv) > 1 else "mixed"

print(f"初始化 GPU 张量... 场景类型: {task_type}")
with torch.cuda.stream(custom_stream):
    a = torch.randn(8192, 8192, device='cuda')
    b = torch.randn(8192, 8192, device='cuda')
    torch.cuda.synchronize()

print(f"开始执行 {task_type} 任务... (按 Ctrl+C 终止)")

batch_count = 0
try:
    while True:
        with torch.cuda.stream(custom_stream):
            start_time = time.time()
            
            if task_type == "interactive":
                # 场景 1: 纯交互式任务 (如短提示词请求)
                # 特点: 计算时间极短, 总是伴随长休眠. 应该永远保持在 Priority 0/1
                for _ in range(10):
                    c = torch.matmul(a, b)
                torch.cuda.synchronize()
                duration = (time.time() - start_time) * 1000
                print(f"[{task_type}] Batch {batch_count} done in {duration:.0f} ms. 思考中...")
                time.sleep(0.3) # 300ms 睡眠, 稳定触发 Priority Recovery (100ms 阈值)

            elif task_type == "background":
                # 场景 2: 纯后台任务 (如模型训练/大文件推理)
                # 特点: 计算时间极长, 几乎无休眠. 应该迅速降级到 Priority 3
                for _ in range(200):
                    c = torch.matmul(a, b)
                torch.cuda.synchronize()
                duration = (time.time() - start_time) * 1000
                print(f"[{task_type}] Batch {batch_count} done in {duration:.0f} ms. 继续计算...")
                time.sleep(0.005) # 5ms 极短睡眠, 无法触发 Recovery

            elif task_type == "mixed":
                # 场景 3: 混合动态任务 (最复杂)
                # 模拟一个会"改变行为模式"的应用:
                # 阶段 A (前 5 个 batch): 后台长生成任务 -> 应该被降级
                # 阶段 B (后 5 个 batch): 突然变为空闲等待用户输入 -> 应该触发 Soft Priority Recovery
                # 不断循环
                
                is_heavy_phase = (batch_count // 5) % 2 == 0
                
                if is_heavy_phase:
                    # 重计算阶段
                    print(f"[{task_type}] [重计算阶段] 开始...")
                    for _ in range(100):
                        c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                    duration = (time.time() - start_time) * 1000
                    print(f"[{task_type}] Batch {batch_count} done in {duration:.0f} ms.")
                    time.sleep(0.01) # 不休息
                else:
                    # 轻计算 + 长休眠阶段
                    print(f"[{task_type}] [轻交互阶段] 开始...")
                    for _ in range(20):
                        c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                    duration = (time.time() - start_time) * 1000
                    print(f"[{task_type}] Batch {batch_count} done in {duration:.0f} ms. 等待用户...")
                    time.sleep(0.5) # 长休眠, 将触发提权

        batch_count += 1

except KeyboardInterrupt:
    print(f"\n{task_type} 任务已终止，共完成 {batch_count} 批次。")
