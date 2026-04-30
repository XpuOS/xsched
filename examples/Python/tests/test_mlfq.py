import torch
import time
import os
import sys

custom_stream = torch.cuda.Stream()

task_type = sys.argv[1] if len(sys.argv) > 1 else "short"

print(f"Initializing GPU tensors... Current task type {task_type}")

with torch.cuda.stream(custom_stream):
    a = torch.randn(8192, 8192, device='cuda')
    b = torch.randn(8192, 8192, device='cuda')
    torch.cuda.synchronize()

print(f"Starting {task_type} task... (Press Ctrl+C to terminate)")

batch_count = 0
try:
    while True:
        with torch.cuda.stream(custom_stream):
            start_time = time.time()
            
            if task_type == "short":
                for _ in range(2):
                    c = torch.matmul(a, b)
                torch.cuda.synchronize() 
                
                duration = (time.time() - start_time) * 1000
                print(f"[{task_type}] Task batch {batch_count} completed in {duration:.0f} ms. Preparing for long sleep (Idle)...")
                
                time.sleep(0.5) 

            elif task_type == "long":
                for _ in range(300):
                    c = torch.matmul(a, b)
                torch.cuda.synchronize()
                
                duration = (time.time() - start_time) * 1000
                print(f"[{task_type}] Task batch {batch_count} completed in {duration:.0f} ms. Brief rest...")
                
                time.sleep(0.01)

        batch_count += 1

except KeyboardInterrupt:
    print(f"\n{task_type} task manually terminated, completed {batch_count} batches.")
