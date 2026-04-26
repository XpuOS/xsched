import torch
import time

print("正在初始化 GPU 张量...")

# 【关键修改】强制创建一个新的 CUDA Stream
# 这会触发底层的 cuStreamCreate API，从而让 XSched 成功拦截并创建受控的 XQueue
custom_stream = torch.cuda.Stream()

with torch.cuda.stream(custom_stream):
    # 创建巨大的矩阵
    a = torch.randn(8192, 8192, device='cuda')
    b = torch.randn(8192, 8192, device='cuda')

    # 预热一下
    torch.matmul(a, b)
    custom_stream.synchronize()

    print("开始执行测试任务... (按 Ctrl+C 终止)")
    batch_idx = 0
    try:
        while True:  # 改为无限循环，直到手动 Ctrl+C
            start_time = time.time()
            
            # 连续做 10 次矩阵乘法
            for _ in range(10):
                torch.matmul(a, b)
            custom_stream.synchronize()  # 等待 GPU 计算完成
            
            end_time = time.time()
            duration_ms = int((end_time - start_time) * 1000)
            print(f"Task batch {batch_idx} completed in {duration_ms} ms")
            batch_idx += 1
            
            time.sleep(0.01) # 稍微休息一下，模拟真实业务的间隙
            
    except KeyboardInterrupt:
        print("\n测试已手动终止。")
