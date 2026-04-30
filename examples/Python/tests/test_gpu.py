import torch
import time

print("Initializing GPU tensors...")

custom_stream = torch.cuda.Stream()

with torch.cuda.stream(custom_stream):
    # Create large matrices
    a = torch.randn(8192, 8192, device='cuda')
    b = torch.randn(8192, 8192, device='cuda')

    # Warm up
    torch.matmul(a, b)
    custom_stream.synchronize()

    print("Starting test task... (Press Ctrl+C to terminate)")
    batch_idx = 0
    try:
        while True:  # Infinite loop until manual Ctrl+C
            start_time = time.time()
            
            # Perform 10 consecutive matrix multiplications
            for _ in range(10):
                torch.matmul(a, b)
            custom_stream.synchronize()
            
            end_time = time.time()
            duration_ms = int((end_time - start_time) * 1000)
            print(f"Task batch {batch_idx} completed in {duration_ms} ms")
            batch_idx += 1
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nTest manually terminated.")
