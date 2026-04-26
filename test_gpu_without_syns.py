import torch
import time

print("正在初始化 GPU 张量...")
custom_stream = torch.cuda.Stream()

# 增加矩阵大小，让单次计算更久，更难被并发
a = torch.randn(8192, 8192, device='cuda')
b = torch.randn(8192, 8192, device='cuda')

with torch.cuda.stream(custom_stream):
    torch.matmul(a, b)
    custom_stream.synchronize()

    print("开始执行测试任务... (按 Ctrl+C 终止)")
    batch_idx = 0
    try:
        while True:
            start_time = time.time()
            
            # 执行 50 次超大矩阵乘法
            for _ in range(50):
                torch.matmul(a, b)
            
            custom_stream.synchronize()  
            
            end_time = time.time()
            duration_ms = int((end_time - start_time) * 1000)
            print(f"Task batch {batch_idx} completed in {duration_ms} ms")
            batch_idx += 1
            
    except KeyboardInterrupt:
        print("\n测试已手动终止。")