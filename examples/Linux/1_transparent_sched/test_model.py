#!/usr/bin/env python3
"""
HIP model inference benchmark for xsched priority scheduling test.

Usage:
    # High priority
    XSCHED_AUTO_XQUEUE_PRIORITY=10 python test_model.py

    # Low priority
    XSCHED_AUTO_XQUEUE_PRIORITY=-10 python test_model.py

Env vars:
    XSCHED_AUTO_XQUEUE_PRIORITY  — priority for xsched (default: 0)
    BATCH_SIZE                   — input batch size (default: 16)
    WARMUP                       — warmup iterations (default: 20)
    ITERS                        — benchmark iterations (default: 500)
    MODEL                        — "resnet50", "resnet101", "simple" (default: resnet50)
"""

import torch
import torch.nn as nn
import time
import os
import sys
import signal
import argparse


def build_resnet(name="resnet50"):
    """Build a ResNet model. Uses torchvision if available, otherwise a simple CNN."""
    try:
        import torchvision.models as models
        if name == "resnet50":
            model = models.resnet50(weights=None)
        elif name == "resnet101":
            model = models.resnet101(weights=None)
        else:
            model = models.resnet50(weights=None)
        model.train(False)
        return model
    except ImportError:
        print("[WARN] torchvision not found, using simple CNN fallback")
        return build_simple_cnn()


def build_simple_cnn():
    """A simple CNN that stresses the GPU enough for scheduling tests."""
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1000),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = SimpleCNN()
    model.train(False)
    return model


def check_device():
    """Check and report HIP/CUDA device info."""
    if not torch.cuda.is_available():
        print("[ERRO] No HIP/CUDA device available!")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"[INFO] Device {i}: {props.name} ({props.total_memory // 1024 // 1024} MiB)")

    return torch.device("cuda:0")


def main():
    parser = argparse.ArgumentParser(description="HIP model inference benchmark")
    parser.add_argument("--model", default=os.environ.get("MODEL", "resnet50"),
                        choices=["resnet50", "resnet101", "simple"])
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "16")))
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP", "20")))
    parser.add_argument("--iters", type=int, default=int(os.environ.get("ITERS", "500")))
    parser.add_argument("--input-size", type=int, default=224)
    args = parser.parse_args()

    priority = os.environ.get("XSCHED_AUTO_XQUEUE_PRIORITY", "default")
    print(f"[INFO] xsched priority: {priority}")
    print(f"[INFO] model: {args.model}, batch_size: {args.batch_size}, "
          f"warmup: {args.warmup}, iters: {args.iters}")

    # Device setup
    device = check_device()

    # Build model
    print(f"[INFO] Loading model '{args.model}'...")
    model = build_resnet(args.model)
    model = model.to(device)

    # Create a non-default stream so xsched can intercept and schedule it.
    # xsched's HipQueueCreate rejects the default (null) stream.
    stream = torch.cuda.Stream()
    print(f"[INFO] Created HIP stream: {stream.cuda_stream}")

    # Fixed random input
    torch.manual_seed(42)
    dummy_input = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)

    # Warmup
    print(f"[INFO] Warming up ({args.warmup} iterations)...")
    with torch.cuda.stream(stream):
        for i in range(args.warmup):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    print("[INFO] Warmup done.")

    # Benchmark
    print(f"[INFO] Benchmarking ({args.iters} iterations)...")
    latencies = []
    start_time = time.time()

    with torch.cuda.stream(stream):
        for i in range(args.iters):
            iter_start = time.time()
            _ = model(dummy_input)
            stream.synchronize()
            iter_end = time.time()
            latencies.append((iter_end - iter_start) * 1000)  # ms

            # Progress report every 100 iters
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                throughput = (i + 1) / elapsed
                avg_lat = sum(latencies[-100:]) / 100
                print(f"[{priority}] iter {i + 1}/{args.iters} | "
                      f"throughput: {throughput:.1f} img/s | "
                      f"avg_latency: {avg_lat:.2f} ms")

    total_time = time.time() - start_time
    avg_latency = sum(latencies) / len(latencies)
    throughput = args.iters / total_time

    print(f"\n[RESULT priority={priority}] "
          f"total_time: {total_time:.2f}s | "
          f"throughput: {throughput:.1f} img/s | "
          f"avg_latency: {avg_latency:.2f} ms | "
          f"batch_size: {args.batch_size} | "
          f"total_images: {args.iters * args.batch_size}")


if __name__ == "__main__":
    main()
