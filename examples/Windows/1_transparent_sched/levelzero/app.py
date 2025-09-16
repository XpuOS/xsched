import torch
import time
import torchvision.models as models
import intel_extension_for_pytorch as ipex

N = 10000  # Number of tasks

def main():
    model = models.resnet152(weights="ResNet152_Weights.DEFAULT")
    model.eval()
    data = torch.rand(1, 3, 224, 224)

    model = model.to("xpu")
    data = data.to("xpu")
    model = ipex.optimize(model)

    with torch.no_grad():
        # warm up
        out = 0
        for _ in range(10):
            output = model(data)
            idx = output[0][0].item()
            out += idx

        for i in range(N):
            begin = time.time()
            output = model(data)
            idx = output[0][0].item()
            out += idx
            end = time.time()
            print("Task %d completed in %.6f seconds" % (i, end - begin))


if __name__ == "__main__":
    main()
