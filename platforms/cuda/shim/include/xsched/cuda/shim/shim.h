#pragma once

#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/common/handle.h"
#include "xsched/cuda/hal/common/cuda_command.h"

namespace xsched::cuda
{

CUstream GetPTDS();
void WaitBlockingXQueues();

void CaptureBegin();
void CaptureEnd();
int64_t GetCaptureCounter();

#define CONVERT_STREAM(stream) \
{ \
    if (stream == nullptr) stream = CU_STREAM_LEGACY; \
    else if (stream == CU_STREAM_PER_THREAD) stream = GetPTDS(); \
}

#define CONVERT_STREAM_PTSZ(stream) \
{ \
    if (stream == nullptr || stream == CU_STREAM_PER_THREAD) stream = GetPTDS(); \
}

/// When there are any streams capturing, all XQueues will be bypassed.
#define CHECK_STREAM_CAPTURE(stream, fallback) \
{ \
    if (GetCaptureCounter() > 0) return fallback; \
}

/// CHECK_STREAM equals to
/// #define CHECK_STREAM(stream, fallback) {
///     CONVERT_STREAM(stream);
///     CHECK_STREAM_CAPTURE(stream, fallback);
///     if (stream == CU_STREAM_LEGACY) { WaitBlockingXQueues(); return fallback; }
/// }
#define CHECK_STREAM(stream, fallback) \
{ \
    if (stream == nullptr || stream == CU_STREAM_LEGACY) { \
        stream = CU_STREAM_LEGACY; \
        if (GetCaptureCounter() == 0) WaitBlockingXQueues(); \
        return fallback; \
    } \
    if (stream == CU_STREAM_PER_THREAD) stream = GetPTDS(); \
    if (GetCaptureCounter() > 0) return fallback; \
}

#define CHECK_STREAM_PTSZ(stream, fallback) \
{ \
    if (stream == CU_STREAM_LEGACY) { \
        if (GetCaptureCounter() == 0) WaitBlockingXQueues(); \
        return fallback; \
    } \
    if (stream == nullptr || stream == CU_STREAM_PER_THREAD) stream = GetPTDS(); \
    if (GetCaptureCounter() > 0) return fallback; \
}

#define CUDA_SHIM_FUNC(name, cmd, ...) \
inline CUresult X##name##Impl(FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__), CUstream stream) \
{ \
    auto xq = xsched::preempt::HwQueueManager::GetXQueue(GetHwQueueHandle(stream)); \
    if (xq == nullptr) return Driver::name(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__), stream); \
    auto hw_cmd = std::make_shared<cmd>(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__)); \
    xq->Submit(hw_cmd); \
    return CUDA_SUCCESS; \
} \
inline CUresult X##name(FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__), CUstream stream) \
{ \
    XDEBG("X" #name "(stream: %p)", stream); \
    CHECK_STREAM(stream, Driver::name(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__), stream)); \
    return X##name##Impl(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__), stream); \
} \
inline CUresult X##name##_ptsz(FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__), CUstream stream) \
{ \
    XDEBG("X" #name "_ptsz(stream: %p)", stream); \
    CHECK_STREAM_PTSZ(stream, Driver::name(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__), stream)); \
    return X##name##Impl(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__), stream); \
}

#define CUDA_GRAPH_CAPTURE_FUNC(name, ...) \
inline CUresult X##name##Impl(CUstream stream __VA_OPT__(,) FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__)) \
{ \
    CaptureBegin(); \
    CUresult res = Driver::name(stream __VA_OPT__(,) FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__)); \
    if (res != CUDA_SUCCESS) CaptureEnd(); \
    return res; \
} \
inline CUresult X##name(CUstream stream __VA_OPT__(,) FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__)) \
{ \
    XDEBG("X" #name "(stream: %p)", stream); \
    CONVERT_STREAM(stream); \
    return X##name##Impl(stream __VA_OPT__(,) FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__)); \
} \
inline CUresult X##name##_ptsz(CUstream stream __VA_OPT__(,) FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__)) \
{ \
    XDEBG("X" #name "_ptsz(stream: %p)", stream); \
    CONVERT_STREAM_PTSZ(stream); \
    return X##name##Impl(stream __VA_OPT__(,) FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__)); \
} \

////////////////////////////// kernel related //////////////////////////////
CUresult XLaunchKernel(CUfunction f, unsigned int gdx, unsigned int gdy, unsigned int gdz, unsigned int bdx, unsigned int bdy, unsigned int bdz, unsigned int shmem, CUstream stream, void **params, void **extra);
CUresult XLaunchKernel_ptsz(CUfunction f, unsigned int gdx, unsigned int gdy, unsigned int gdz, unsigned int bdx, unsigned int bdy, unsigned int bdz, unsigned int shmem, CUstream stream, void **params, void **extra);
CUresult XLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **params, void **extra);
CUresult XLaunchKernelEx_ptsz(const CUlaunchConfig *config, CUfunction f, void **params, void **extra);
CUresult XLaunchHostFunc(CUstream stream, CUhostFn fn, void *data);
CUresult XLaunchHostFunc_ptsz(CUstream stream, CUhostFn fn, void *data);

////////////////////////////// graph related //////////////////////////////
CUDA_GRAPH_CAPTURE_FUNC(StreamBeginCapture);
CUDA_GRAPH_CAPTURE_FUNC(StreamBeginCapture_v2, CUstreamCaptureMode, mode);
CUDA_GRAPH_CAPTURE_FUNC(StreamBeginCaptureToGraph, CUgraph, hGraph, const CUgraphNode *, dependencies, const CUgraphEdgeData *, dependencyData, size_t, numDependencies, CUstreamCaptureMode, mode);
CUresult XStreamEndCapture(CUstream stream, CUgraph *graph);
CUresult XStreamEndCapture_ptsz(CUstream stream, CUgraph *graph);

CUDA_SHIM_FUNC(GraphUpload, CudaGraphUploadCommand, CUgraphExec, graph_exec);
CUDA_SHIM_FUNC(GraphLaunch, CudaGraphLaunchCommand, CUgraphExec, graph_exec);

////////////////////////////// memory related //////////////////////////////
CUDA_SHIM_FUNC(MemcpyHtoDAsync_v2, CudaMemcpyHtoDV2Command, CUdeviceptr, dstDevice, const void *, srcHost, size_t, ByteCount);
CUDA_SHIM_FUNC(MemcpyDtoHAsync_v2, CudaMemcpyDtoHV2Command, void *, dstHost, CUdeviceptr, srcDevice, size_t, ByteCount);
CUDA_SHIM_FUNC(MemcpyDtoDAsync_v2, CudaMemcpyDtoDV2Command, CUdeviceptr, dstDevice, CUdeviceptr, srcDevice, size_t, ByteCount);
CUDA_SHIM_FUNC(Memcpy2DAsync_v2, CudaMemcpy2DV2Command, const CUDA_MEMCPY2D *, pCopy);
CUDA_SHIM_FUNC(Memcpy3DAsync_v2, CudaMemcpy3DV2Command, const CUDA_MEMCPY3D *, pCopy);
CUDA_SHIM_FUNC(MemsetD8Async, CudaMemsetD8Command, CUdeviceptr, dstDevice, unsigned char, uc, size_t, N);
CUDA_SHIM_FUNC(MemsetD16Async, CudaMemsetD16Command, CUdeviceptr, dstDevice, unsigned short, us, size_t, N);
CUDA_SHIM_FUNC(MemsetD32Async, CudaMemsetD32Command, CUdeviceptr, dstDevice, unsigned int, ui, size_t, N);
CUDA_SHIM_FUNC(MemsetD2D8Async, CudaMemsetD2D8Command, CUdeviceptr, dstDevice, size_t, dstPitch, unsigned char, uc, size_t, Width, size_t, Height);
CUDA_SHIM_FUNC(MemsetD2D16Async, CudaMemsetD2D16Command, CUdeviceptr, dstDevice, size_t, dstPitch, unsigned short, us, size_t, Width, size_t, Height);
CUDA_SHIM_FUNC(MemsetD2D32Async, CudaMemsetD2D32Command, CUdeviceptr, dstDevice, size_t, dstPitch, unsigned int, ui, size_t, Width, size_t, Height);
CUDA_SHIM_FUNC(MemFreeAsync, CudaMemoryFreeCommand, CUdeviceptr, dptr);
CUDA_SHIM_FUNC(MemAllocAsync, CudaMemoryAllocCommand, CUdeviceptr *, dptr, size_t, bytesize);

CUresult XMemFree_v2(CUdeviceptr dptr);

////////////////////////////// event related //////////////////////////////
CUresult XEventRecord(CUevent event, CUstream stream);
CUresult XEventRecord_ptsz(CUevent event, CUstream stream);
CUresult XEventRecordWithFlags(CUevent event, CUstream stream, unsigned int flags);
CUresult XEventRecordWithFlags_ptsz(CUevent event, CUstream stream, unsigned int flags);
CUresult XEventQuery(CUevent event);
CUresult XEventSynchronize(CUevent event);
CUresult XStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags);
CUresult XStreamWaitEvent_ptsz(CUstream stream, CUevent event, unsigned int flags);
CUresult XEventDestroy(CUevent event);
CUresult XEventDestroy_v2(CUevent event);

////////////////////////////// stream related //////////////////////////////
CUresult XStreamSynchronize(CUstream stream);
CUresult XStreamSynchronize_ptsz(CUstream stream);
CUresult XStreamQuery(CUstream stream);
CUresult XStreamQuery_ptsz(CUstream stream);
CUresult XCtxSynchronize();

CUresult XStreamCreate(CUstream *stream, unsigned int flags);
CUresult XStreamCreateWithPriority(CUstream *stream, unsigned int flags, int priority);
CUresult XStreamDestroy(CUstream stream);
CUresult XStreamDestroy_v2(CUstream stream);

} // namespace xsched::cuda
