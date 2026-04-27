import torch
import time
import os
import sys

# 必须显式创建一个 Custom Stream，否则 xsched 无法拦截并区分任务
custom_stream = torch.cuda.Stream()

# 获取用户传入的任务类型（默认是 short）
task_type = sys.argv[1] if len(sys.argv) > 1 else "short"

print(f"初始化 GPU 张量... 当前任务类型: {task_type}")
# 预热并分配内存
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
            
            # 短任务：快速执行完一小批矩阵乘法，然后 Sleep 很久 (模拟交互式响应，比如短 Prompt)
            if task_type == "short":
                for _ in range(2):
                    c = torch.matmul(a, b)
                torch.cuda.synchronize() # 确保执行完成
                
                duration = (time.time() - start_time) * 1000
                print(f"[{task_type}] Task batch {batch_count} completed in {duration:.0f} ms. 准备进入长休眠(Idle)...")
                
                # 关键：Idle 时间要大于我们在 mlfq.cpp 里设置的 recovery_threshold_ (100ms)
                time.sleep(0.5) 

            # 长任务：疯狂进行矩阵乘法，不怎么休息 (模拟后台长文本生成、模型训练)
            elif task_type == "long":
                for _ in range(300):
                    c = torch.matmul(a, b)
                torch.cuda.synchronize()
                
                duration = (time.time() - start_time) * 1000
                print(f"[{task_type}] Task batch {batch_count} completed in {duration:.0f} ms. 短暂休息...")
                
                # 只有非常短暂的间隔，不足以触发 Priority Recovery
                time.sleep(0.01)

        batch_count += 1

except KeyboardInterrupt:
    print(f"\n{task_type} 任务已手动终止，共完成 {batch_count} 批次。")
