import sys
import time
import requests
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# 应用团队提供的真实业务数据
REAL_DOCUMENTS = [
    "深度学习是机器学习的一个分支，它基于人工神经网络，特别是多层非线性处理单元的层次结构...",
    "卷积神经网络（CNN）是专门设计用于处理具有网格结构数据的深度学习模型，如图像...",
    "目标检测是计算机视觉中的核心任务，旨在识别图像中所有感兴趣的目标并确定其位置...",
    "图像分割任务要求将图像划分为具有语义意义的区域。语义分割为每个像素分配类别标签...",
    "生成对抗网络（GAN）由生成器和判别器两个神经网络组成，通过对抗训练学习数据分布...",
    "注意力机制使神经网络能够动态关注输入的不同部分，显著提升了模型性能...",
    "迁移学习利用在源任务上预训练的模型来解决目标任务，特别适用于数据稀缺的场景...",
    "数据增强通过对训练数据进行变换来扩充数据集，提高模型的泛化能力...",
    "神经网络架构搜索（NAS）旨在自动发现最优的网络结构，减少人工设计的工作量...",
    "模型压缩技术减小深度学习模型的存储和计算需求，使其适用于资源受限环境...",
    "视频理解任务需要处理时间维度信息，分析动态场景中的动作和事件...",
    "人体姿态估计旨在定位图像或视频中人体关键点的位置，如关节和面部特征...",
    "人脸识别技术通过分析面部特征验证或识别个人身份。传统方法使用手工设计的局部特征描述子...",
    "光学字符识别（OCR）将图像中的文本转换为机器可读格式。传统 OCR 流程包括文本检测...",
    "医学图像分析利用深度学习辅助疾病诊断和治疗规划。卷积神经网络用于病灶检测...",
    "自动驾驶感知系统依赖计算机视觉理解周围环境。目标检测识别车辆、行人和交通标志...",
    "图像超分辨率从低分辨率输入重建高分辨率图像，提升视觉质量和细节清晰度...",
    "图像风格迁移将艺术作品的视觉风格应用到内容图像上，创造具有艺术效果的图像...",
    "自监督学习利用数据本身的结构生成监督信号，无需人工标注。对比学习通过拉近相似样本...",
    "神经辐射场（NeRF）使用全连接神经网络隐式表示三维场景，从多视角图像合成新视角..."
]

def online_request(task, port, query, documents, vllm=False, model="rerank-english-v3.0", model_type="bge"):
    # 【核心修改区】：为 Qwen3-VL 补全完整的多模态视觉标签
    if model_type == "qwen-vl":
        pad_str = "<|vision_start|><|image_pad|><|vision_end|>"
        
        # 清理可能残留的旧标签，防止重复添加导致 Attention 错位
        query_clean = query.replace("<|image_pad|>", "").replace("<|vision_start|>", "").replace("<|vision_end|>", "")
        query = pad_str + query_clean
        
        docs_clean = []
        for doc in documents:
            d_clean = doc.replace("<|image_pad|>", "").replace("<|vision_start|>", "").replace("<|vision_end|>", "")
            docs_clean.append(pad_str + d_clean)
        documents = docs_clean

    if vllm:
        # SGLang 完美兼容 /v1/rerank 接口标准
        url = f"http://127.0.0.1:{port}/v1/rerank"
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": 20
        }
    else:
        url = f"http://127.0.0.1:{port}/v1/predict"
        payload = {
            "task": task,
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": 20,
            "return_documents": True
        }
    
    try:
        start = time.perf_counter()
        r = requests.post(url, json=payload, timeout=20)
        
        # 降级处理逻辑
        if vllm and r.status_code == 404:
            url_emb = f"http://127.0.0.1:{port}/v1/embeddings"
            emb_payload = {
                "model": model,
                "input": [query] + documents
            }
            r = requests.post(url_emb, json=emb_payload, timeout=20)
            
        end = time.perf_counter()
        if r.status_code == 200:
            return (end - start) * 1000 # ms
        else:
            print(f"❌ 请求报错 (HTTP {r.status_code}): {r.text}")
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Online Rerank Benchmark with Real Data")
    parser.add_argument("--num", type=int, default=50, help="Total requests")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrency")
    
    # 将默认参数调整为直接匹配 SGLang 的配置
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--vllm", action="store_true", help="Use standard vLLM/SGLang Rerank API")
    parser.add_argument("--model", type=str, default="qwen3-vl-embedding-2b", help="Model name")
    parser.add_argument("--model-type", type=str, default="qwen-vl", choices=["bge", "qwen-vl"], help="Model type for padding")
    args = parser.parse_args()

    query = "深度学习在计算机视觉领域的应用与发展"

    print(f"🚀 Starting REAL-WORLD Rerank Online Benchmark")
    print(f"🚀 Query: {query}")
    print(f"🚀 Documents: {len(REAL_DOCUMENTS)} (High Load)")
    print(f"🚀 Sending {args.num} requests with concurrency {args.concurrency}...")
    print(f"🚀 Model Type: {args.model_type}")
    
    if args.vllm:
        print(f"🚀 Mode: SGLang/vLLM API (Model: {args.model}, Port: {args.port})")
    else:
        print(f"🚀 Mode: Predict API (Model: {args.model}, Port: {args.port})")

    latencies = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(online_request, "rerank", args.port, query, REAL_DOCUMENTS, args.vllm, args.model, args.model_type) for _ in range(args.num)]
        for f in futures:
            res = f.result()
            if res:
                latencies.append(res)

    if latencies:
        print("\n📊 ONLINE PERFORMANCE (Rerank @ 20 Docs):")
        print("-" * 45)
        print(f"Avg Latency: {np.mean(latencies):.2f} ms")
        print(f"P50 Latency: {np.percentile(latencies, 50):.2f} ms")
        print(f"P99 Latency: {np.percentile(latencies, 99):.2f} ms")
        print(f"Throughput:  {len(latencies)/(np.sum(latencies)/1000/args.concurrency):.2f} RPS")
        print("-" * 45)
    else:
        print("❌ All requests failed.")

if __name__ == "__main__":
    main()