import torch
import time
import os
import sys
import random

custom_stream = torch.cuda.Stream()
task_type = sys.argv[1] if len(sys.argv) > 1 else "mixed"

print(f"Initializing GPU tensors... Scenario type: {task_type}")
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
            
            if task_type == "interactive":
                # Scenario 1: Pure interactive task (e.g., short prompt request)
                for _ in range(10):
                    c = torch.matmul(a, b)
                torch.cuda.synchronize()
                duration = (time.time() - start_time) * 1000
                print(f"[{task_type}] Batch {batch_count} done in {duration:.0f} ms. Thinking...")
                time.sleep(0.3) # 300ms sleep, stably triggers Priority Recovery (100ms threshold)

            elif task_type == "background":
                # Scenario 2: Pure background task (e.g., model training/large file inference)
                for _ in range(200):
                    c = torch.matmul(a, b)
                torch.cuda.synchronize()
                duration = (time.time() - start_time) * 1000
                print(f"[{task_type}] Batch {batch_count} done in {duration:.0f} ms. Computing...")
                time.sleep(0.005) # 5ms extremely short sleep, cannot trigger Recovery

            elif task_type == "mixed":
                # Scenario 3: Mixed dynamic task (Most complex)
                
                is_heavy_phase = (batch_count // 5) % 2 == 0
                
                if is_heavy_phase:
                    # Heavy computation phase
                    print(f"[{task_type}] [Heavy Comp Phase] Start...")
                    for _ in range(100):
                        c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                    duration = (time.time() - start_time) * 1000
                    print(f"[{task_type}] Batch {batch_count} done in {duration:.0f} ms.")
                    time.sleep(0.01) 
                else:
                    # Light computation + Long sleep phase
                    print(f"[{task_type}] [Light Interactive Phase] Start...")
                    for _ in range(20):
                        c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                    duration = (time.time() - start_time) * 1000
                    print(f"[{task_type}] Batch {batch_count} done in {duration:.0f} ms. Waiting for user...")
                    time.sleep(0.5)

        batch_count += 1

except KeyboardInterrupt:
    print(f"\n{task_type} task terminated, completed {batch_count} batches.")
