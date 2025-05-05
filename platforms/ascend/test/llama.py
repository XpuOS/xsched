import torch_npu
import transformers
import torch
from transformers import TextIteratorStreamer
from threading import Thread
import argparse

def run_inference(model_path, prompt, max_new_tokens, npu):
    # Specify a single NPU (device 0)
    device = f"npu:{npu}"  # Use NPU device 0

    # Create the pipeline with the local model path and specific device
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=device
    )

    # Create a streamer
    streamer = TextIteratorStreamer(pipeline.tokenizer)

    # Generate in a separate thread
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "streamer": streamer,
    }

    # Start generation in a separate thread
    thread = Thread(target=pipeline, args=[prompt], kwargs=generation_kwargs)
    thread.start()

    # Iterate over the generated text
    print("Streaming output:")
    for text in streamer:
        print(text, end="", flush=True)  # Print each chunk as it's generated

    # Wait for the generation to finish
    thread.join()
    print("\nGeneration complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, default="/home/ma-user/modelarts/inputs/model_path_0/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/74fbf131a939963dd1e244389bb61ad0d0440a4d/")
    parser.add_argument("--prompt", "-p", type=str, default="Hey how are you doing today?")
    parser.add_argument("--max_new_tokens", "-n", type=int, default=4096)
    parser.add_argument("--npu", "-d", type=int, default=0)
    args = parser.parse_args()

    run_inference(args.model_path, args.prompt, args.max_new_tokens, args.npu)
