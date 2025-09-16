import sys
import time
import numpy as np
import openvino as ov

infer_cnt = 1000
core = ov.Core()

model = core.read_model(model=sys.argv[1])
model = core.compile_model(model, "NPU", config={"NPU_USE_NPUW":"YES"})

input = np.ones((1, 3, 224, 224), dtype=np.float32)

request = model.create_infer_request()
request.start_async(input)
request.wait()

print("inference start", file=sys.stderr, flush=True)

request.start_async(input)
request.wait()

print("inference done", file=sys.stderr, flush=True)

output = request.get_output_tensor().data
predicted_idx = np.argmax(output)
print(f'predicted class: {predicted_idx}', file=sys.stderr, flush=True)

print("warmup...", file=sys.stderr, flush=True)
for _ in range(infer_cnt):
    request.start_async(input)
    request.wait()

print("benchmark...", file=sys.stderr, flush=True)
start = time.time()
for _ in range(infer_cnt):
    request.start_async(input)
    request.wait()
end = time.time()
print(f"inference time: {(end - start) * 1000 / infer_cnt:.2f} ms", file=sys.stderr, flush=True)

output = request.get_output_tensor().data
predicted_idx = np.argmax(output)
print(f'predicted class: {predicted_idx}', file=sys.stderr, flush=True)
