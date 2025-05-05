#pragma once

#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/common/handle.h"
#include "xsched/cuda/hal/common/cuda_command.h"

namespace xsched::cuda
{

#define CUDA_SHIM_FUNC(name, cmd, ...) \
inline CUresult X##name(FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__), CUstream stream) \
{ \
    if (stream == 0) { \
        WaitBlockingXQueues(); \
        return Driver::name(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__), stream); \
    } \
    auto xq = xsched::preempt::HwQueueManager::GetXQueue(GetHwQueueHandle(stream)); \
    if (xq == nullptr) return Driver::name(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__), stream); \
    auto hw_cmd = std::make_shared<cmd>(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__)); \
    xq->Submit(hw_cmd); \
    return CUDA_SUCCESS; \
}

void WaitBlockingXQueues();

////////////////////////////// kernel related //////////////////////////////
CUresult XLaunchKernel(CUfunction f, unsigned int gdx, unsigned int gdy, unsigned int gdz, unsigned int bdx, unsigned int bdy, unsigned int bdz, unsigned int shmem, CUstream stream, void **params, void **extra);
CUresult XLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **params, void **extra);
CUresult XLaunchHostFunc(CUstream stream, CUhostFn fn, void *data);

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

////////////////////////////// event related //////////////////////////////
CUresult XEventQuery(CUevent event);
CUresult XEventRecord(CUevent event, CUstream stream);
CUresult XEventRecordWithFlags(CUevent event, CUstream stream, unsigned int flags);
CUresult XEventSynchronize(CUevent event);
CUresult XStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags);
CUresult XEventDestroy(CUevent event);
CUresult XEventDestroy_v2(CUevent event);

////////////////////////////// stream related //////////////////////////////
CUresult XStreamSynchronize(CUstream stream);
CUresult XStreamQuery(CUstream stream);
CUresult XCtxSynchronize();

CUresult XStreamCreate(CUstream *stream, unsigned int flags);
CUresult XStreamCreateWithPriority(CUstream *stream, unsigned int flags, int priority);
CUresult XStreamDestroy(CUstream stream);
CUresult XStreamDestroy_v2(CUstream stream);

} // namespace xsched::cuda
