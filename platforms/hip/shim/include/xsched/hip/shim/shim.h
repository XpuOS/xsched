#pragma once

#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/hip/hal/handle.h"
#include "xsched/hip/hal/driver.h"
#include "xsched/hip/hal/hip_command.h"

namespace xsched::hip
{

#define HIP_SHIM_FUNC(name, cmd, ...) \
inline hipError_t X##name(FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__), hipStream_t stream) \
{ \
    if (stream == 0) { \
        HipSyncBlockingXQueues(); \
        return Driver::name(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__), stream); \
    } \
    auto xq = xsched::preempt::HwQueueManager::GetXQueue(GetHwQueueHandle(stream)); \
    if (xq == nullptr) return Driver::name(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__), stream); \
    auto hw_cmd = std::make_shared<cmd>(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__)); \
    xq->Submit(hw_cmd); \
    return hipSuccess; \
}

void HipSyncBlockingXQueues();

////////////////////////////// kernel related //////////////////////////////
hipError_t XLaunchKernel(const void *f, dim3 numBlocks, dim3 dimBlocks, void **args, size_t sharedMemBytes, hipStream_t stream);
hipError_t XModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void **kernel_params, void **extra_params);
hipError_t XExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX, uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ, uint32_t localWorkSizeX, uint32_t localWorkSizeY, uint32_t localWorkSizeZ, size_t sharedMemBytes, hipStream_t hStream, void** kernelParams, void** extra, hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags);

void** XRegisterFatBinary(const void* data);
void XRegisterFunction(void** modules, const void* hostFunction, char* deviceFunction, const char* deviceName, unsigned int threadLimit, void* tid, void* bid, dim3* blockDim, dim3* gridDim, int* wSize);

////////////////////////////// memory related //////////////////////////////
// HIP_SHIM_FUNC(MemcpyWithStream, HipMemcpyWithStreamCommand, void *, dst, const void *, src, size_t, sizeBytes,hipMemcpyKind, kind);
HIP_SHIM_FUNC(MemcpyHtoDAsync, HipMemcpyHtoDCommand, hipDeviceptr_t, dst_dev, void *, src_host, size_t, byte_cnt);
HIP_SHIM_FUNC(MemcpyDtoHAsync, HipMemcpyDtoHCommand, void *, dst_host, hipDeviceptr_t, src_dev, size_t, byte_cnt);
HIP_SHIM_FUNC(MemcpyDtoDAsync, HipMemcpyDtoDCommand, hipDeviceptr_t, dst_dev, hipDeviceptr_t, src_dev, size_t, byte_cnt);
// HIP_SHIM_FUNC(MemcpyAsync, HipMemcpyAsyncCommand, void *, dst, const void *, src, size_t, sizeBytes, hipMemcpyKind, kind);
HIP_SHIM_FUNC(MemsetAsync, HipMemsetAsyncCommand, void *, dst, int, value, size_t, sizeBytes);
HIP_SHIM_FUNC(MemsetD8Async, HipMemsetD8Command, hipDeviceptr_t, dst_dev, unsigned char, uc, size_t, n);
HIP_SHIM_FUNC(MemsetD16Async, HipMemsetD16Command, hipDeviceptr_t, dst_dev, unsigned short, us, size_t, n);
HIP_SHIM_FUNC(MemsetD32Async, HipMemsetD32Command, hipDeviceptr_t, dst_dev, unsigned int, ui, size_t, n);

hipError_t XMalloc(void **ptr, size_t size);
hipError_t XFree(void *ptr);
hipError_t XMemcpyAsync(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream);
hipError_t XMemcpyWithStream(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream);

////////////////////////////// event related //////////////////////////////
hipError_t XEventQuery(hipEvent_t event);
hipError_t XEventRecord(hipEvent_t event, hipStream_t stream);
hipError_t XEventRecordWithFlags(hipEvent_t event, hipStream_t stream, unsigned int flags);
hipError_t XEventSynchronize(hipEvent_t event);
hipError_t XStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags);
hipError_t XEventDestroy(hipEvent_t event);

////////////////////////////// stream related //////////////////////////////
hipError_t XStreamSynchronize(hipStream_t stream);
hipError_t XStreamQuery(hipStream_t stream);
hipError_t XCtxSynchronize();

hipError_t XStreamCreate(hipStream_t *stream);
hipError_t XStreamCreateWithFlags(hipStream_t *stream, unsigned int flags);
hipError_t XStreamCreateWithPriority(hipStream_t *stream, unsigned int flags, int priority);
hipError_t XStreamDestroy(hipStream_t stream);

} // namespace xsched::shim::hip
