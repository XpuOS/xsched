#pragma once

#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/corex/hal/common/driver.h"
#include "xsched/corex/hal/common/handle.h"
#include "xsched/corex/hal/common/corex_command.h"

namespace xsched::corex
{

#define COREXRT_SHIM_FUNC(name, cmd, ptsz, ...) \
inline cudaError_t XRt##name(FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__), cudaStream_t stream) \
{ \
    if (stream == nullptr || stream == cudaStreamLegacy) { \
        WaitBlockingXQueues(); \
        return RtDriver::name(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__), stream); \
    } \
    auto xq = xsched::preempt::HwQueueManager::GetXQueue(GetHwQueueHandle(stream)); \
    if (xq == nullptr) return RtDriver::name(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__), stream); \
    auto hw_cmd = std::make_shared<cmd>(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__), ptsz); \
    xq->Submit(hw_cmd); \
    return cudaSuccess; \
}

void WaitBlockingXQueues();

////////////////////////////// kernel related //////////////////////////////
cudaError_t XRtLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
cudaError_t XRtLaunchKernel_ptsz(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
cudaError_t XRtLaunchKernelExC(const cudaLaunchConfig_t *config, const void *func, void **args);

////////////////////////////// memory related //////////////////////////////
COREXRT_SHIM_FUNC(MemcpyAsync     , CorexMemcpyCommand, false, void *, dst, const void *, src, size_t, count, cudaMemcpyKind, kind);
COREXRT_SHIM_FUNC(MemcpyAsync_ptsz, CorexMemcpyCommand, true , void *, dst, const void *, src, size_t, count, cudaMemcpyKind, kind);
COREXRT_SHIM_FUNC(Memcpy2DAsync     , CorexMemcpy2DCommand, false, void *, dst, size_t, dpitch, const void *, src, size_t, spitch, size_t, width, size_t, height, cudaMemcpyKind, kind);
COREXRT_SHIM_FUNC(Memcpy2DAsync_ptsz, CorexMemcpy2DCommand, true , void *, dst, size_t, dpitch, const void *, src, size_t, spitch, size_t, width, size_t, height, cudaMemcpyKind, kind);
COREXRT_SHIM_FUNC(Memcpy3DAsync     , CorexMemcpy3DCommand, false, const cudaMemcpy3DParms *, p);
COREXRT_SHIM_FUNC(Memcpy3DAsync_ptsz, CorexMemcpy3DCommand, true , const cudaMemcpy3DParms *, p);
COREXRT_SHIM_FUNC(MemsetAsync     , CorexMemsetCommand, false, void *, devPtr, int, value, size_t, count);
COREXRT_SHIM_FUNC(MemsetAsync_ptsz, CorexMemsetCommand, true , void *, devPtr, int, value, size_t, count);

cudaError_t XRtFree(void *devPtr);

////////////////////////////// event related //////////////////////////////
cudaError_t XRtEventQuery(cudaEvent_t event);
cudaError_t XRtEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t XRtEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream);
cudaError_t XRtEventSynchronize(cudaEvent_t event);
cudaError_t XRtStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
cudaError_t XRtStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
cudaError_t XRtEventDestroy(cudaEvent_t event);

////////////////////////////// stream related //////////////////////////////
cudaError_t XRtDeviceSynchronize();

CUresult XStreamSynchronize(CUstream hStream);
CUresult XStreamSynchronize_ptsz(CUstream hStream);
cudaError_t XRtStreamSynchronize(cudaStream_t stream);
cudaError_t XRtStreamSynchronize_ptsz(cudaStream_t stream);
cudaError_t XRtStreamQuery(cudaStream_t stream);
cudaError_t XRtStreamQuery_ptsz(cudaStream_t stream);

cudaError_t XRtStreamCreate(cudaStream_t *stream);
cudaError_t XRtStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags);
cudaError_t XRtStreamCreateWithPriority(cudaStream_t *stream, unsigned int flags, int priority);
cudaError_t XRtStreamDestroy(cudaStream_t stream);

CUresult XStreamCreate(CUstream *phStream, unsigned int flags);
CUresult XStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority);
CUresult XStreamDestroy_v2(CUstream hStream);

} // namespace xsched::corex
