import os
import time
import argparse
import threading
import ctypes
import torch
import uvicorn
import sys
import io
import base64
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

app = FastAPI()
model_obj = None
processor_obj = None # CLIP 需要 processor
stream_factory = None
device = torch.device("cuda:0")
CONFIG = {"prio": 10, "task": ""}

# --- XSched 流工厂 ---
class XSchedStreamFactory:
    def __init__(self, shim_path):
        self.shim_path = shim_path
        self.env_lock = threading.Lock()
        self.stream_cache = {} 
        if self.shim_path and os.path.exists(self.shim_path):
            try:
                self.libshim = ctypes.CDLL(self.shim_path)
            except:
                self.libshim = None
        else:
            self.libshim = None

    def get_stream(self, priority: int) -> torch.cuda.Stream:
        if os.environ.get("XSCHED_ENABLED") == "OFF" or not self.libshim:
            return torch.cuda.current_stream()
        if priority in self.stream_cache:
            return self.stream_cache[priority]
        
        _ = torch.cuda.current_stream() 
        with self.env_lock:
            os.environ["XSCHED_AUTO_XQUEUE_PRIORITY"] = str(priority)
            try:
                stream_ptr = ctypes.c_void_p()
                ret = self.libshim.cuStreamCreate(ctypes.byref(stream_ptr), 0)
                if ret != 0: return torch.cuda.current_stream()
                stream = torch.cuda.ExternalStream(stream_ptr.value)
                self.stream_cache[priority] = stream
                return stream
            except:
                return torch.cuda.current_stream()

# --- 请求结构 ---
class Qwen3VLRequest(BaseModel):
    task: Optional[str] = None
    inputs: Optional[List[Dict[str, Any]]] = None 
    query: Optional[Union[str, Dict[str, Any]]] = None
    documents: Optional[List[Union[str, Dict[str, Any]]]] = None
    instruction: str = "Represent the user's input."
    # 缝合自 full_mix
    text: Optional[str] = None
    image_base64: Optional[str] = None

def decode_image(b64_str):
    try:
        image_data = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        return img.resize((224, 224))
    except:
        return Image.new('RGB', (224, 224), color='white')

@app.post("/v1/predict")
async def predict(req: Qwen3VLRequest):
    stream = stream_factory.get_stream(CONFIG["prio"])
    task = req.task or CONFIG["task"]
    start_time = time.perf_counter()
    
    try:
        with torch.cuda.stream(stream):
            if task == "embedding":
                if not req.inputs:
                    raise HTTPException(status_code=400, detail="Missing 'inputs' for embedding")
                with torch.no_grad():
                    embeddings = model_obj.process(req.inputs)
                stream.synchronize()
                return {"embeddings": embeddings.tolist(), "latency_ms": (time.perf_counter()-start_time)*1000}

            elif task == "rerank":
                if not req.query or not req.documents:
                    raise HTTPException(status_code=400, detail="Missing query/documents")
                q = {"text": req.query} if isinstance(req.query, str) else req.query
                docs = [{"text": d} if isinstance(d, str) else d for d in req.documents]
                with torch.no_grad():
                    scores = model_obj.process({"instruction": req.instruction, "query": q, "documents": docs})
                    if isinstance(scores, torch.Tensor):
                        scores = scores.float().cpu().tolist()
                stream.synchronize()
                return {"scores": scores, "latency_ms": (time.perf_counter()-start_time)*1000}

            elif task == "clip":
                # 缝合自 full_mix
                img = decode_image(req.image_base64) if req.image_base64 else Image.new('RGB', (224, 224), color='red')
                inputs = processor_obj(text=[req.text or "image"], images=img, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model_obj(**inputs)
                    # 默认返回 image_embeds
                    probs = outputs.logits_per_image.softmax(dim=-1).cpu().tolist()
                stream.synchronize()
                return {"probs": probs, "latency_ms": (time.perf_counter()-start_time)*1000}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def load_model(args):
    global model_obj, processor_obj, stream_factory
    
    CONFIG["task"] = args.task
    CONFIG["prio"] = args.default_priority
    
    print(f"📥 Loading Task: {args.task} from {args.model_path} (Target Prio: {args.default_priority}) ...")

    try:
        if args.task == "clip":
            model_obj = ChineseCLIPModel.from_pretrained(args.model_path).to(device)
            processor_obj = ChineseCLIPProcessor.from_pretrained(args.model_path)
            model_obj.eval()
        else:
            # Qwen3-VL 专用加载逻辑
            if args.model_path not in sys.path:
                sys.path.append(args.model_path)
            attn_impl = "eager"
            
            if args.task == "embedding":
                from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
                model_obj = Qwen3VLEmbedder(
                    model_name_or_path=args.model_path, 
                    dtype=torch.bfloat16,
                    attn_implementation=attn_impl
                )
            else:
                from scripts.qwen3_vl_reranker import Qwen3VLReranker
                model_obj = Qwen3VLReranker(
                    model_name_or_path=args.model_path, 
                    dtype=torch.bfloat16,
                    attn_implementation=attn_impl
                )
        print(f"✅ {args.task} model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    stream_factory = XSchedStreamFactory(args.shim_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["embedding", "rerank", "clip"])
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--shim-path", type=str, default="")
    parser.add_argument("--default-priority", type=int, default=10)
    args = parser.parse_args()
    load_model(args)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
