import sys
import time
import torch
import torchvision
import torch_npu
import argparse

BATCH_SIZE = 32

def infer(model, input):
    with torch.no_grad():
        return model(input).cpu()

def run(run_cnt):
    model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT)
    model.eval().npu()
    input = torch.ones(BATCH_SIZE, 3, 224, 224).npu()
    print(infer(model, input))
    
    while True:
        start = time.time()
        for i in range(run_cnt):
            infer(model, input)
        end = time.time()
        print(f"thpt: {BATCH_SIZE * run_cnt / (end - start):.2f} img/s")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="ResNet152 inference on Ascend NPU")
    argparse.add_argument("-c", "--run-cnt", type=int, default=10, help="Run count for inference")
    args = argparse.parse_args()
    run_cnt = args.run_cnt
    run(run_cnt)
