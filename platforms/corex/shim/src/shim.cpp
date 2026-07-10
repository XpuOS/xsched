#include "xsched/utils/map.h"
#include "xsched/utils/xassert.h"
#include "xsched/corex/hal.h"
#include "xsched/corex/shim/shim.h"
#include "xsched/corex/hal/level1/corex_queue.h"
#include "xsched/corex/hal/common/corex_command.h"

using namespace xsched::preempt;

namespace xsched::corex
{

static utils::ObjectMap<cudaEvent_t, std::shared_ptr<CorexEventRecordCommand>> g_events;

void WaitBlockingXQueues()
{
    std::list<std::shared_ptr<XQueueWaitAllCommand>> wait_cmds;
    XResult res = XQueueManager::ForEach([&](std::shared_ptr<XQueue> xq)->XResult {
        auto hwq = xq->GetHwQueue();
        auto corex_q = std::dynamic_pointer_cast<CorexQueueLv1>(hwq);
        if (corex_q == nullptr) return kXSchedErrorUnknown;
        // does not need to wait a non-blocking stream
        if (corex_q->GetStreamFlags() & cudaStreamNonBlocking) return kXSchedSuccess;
        auto wait_cmd = xq->SubmitWaitAll();
        if (wait_cmd == nullptr) return kXSchedErrorUnknown;
        wait_cmds.push_back(wait_cmd);
        return kXSchedSuccess;
    });
    XASSERT(res == kXSchedSuccess, "Fail to submit wait all commands");
    for (auto &cmd : wait_cmds) cmd->Wait();
}

static cudaError_t SubmitKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                void **args, size_t sharedMem, cudaStream_t stream, bool ptsz)
{
    if (stream == nullptr || stream == cudaStreamLegacy) {
        WaitBlockingXQueues();
        return COREXRT_LAUNCH(ptsz, LaunchKernel, func, gridDim, blockDim, args, sharedMem, stream);
    }

    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) {
        return COREXRT_LAUNCH(ptsz, LaunchKernel, func, gridDim, blockDim, args, sharedMem, stream);
    }

    auto kernel = std::make_shared<CorexKernelLaunchCommand>(
        func, gridDim, blockDim, args, sharedMem, xq != nullptr, ptsz);
    xq->Submit(kernel);
    return cudaSuccess;
}

cudaError_t XRtLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream)
{
    XDEBG("XRtLaunchKernel(func: %p, stream: %p, grid: [%u, %u, %u], block: [%u, %u, %u], "
          "shm: %zu, params: %p)", func, stream, gridDim.x, gridDim.y, gridDim.z,
          blockDim.x, blockDim.y, blockDim.z, sharedMem, args);
    return SubmitKernel(func, gridDim, blockDim, args, sharedMem, stream, false);
}

cudaError_t XRtLaunchKernel_ptsz(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream)
{
    XDEBG("XRtLaunchKernel_ptsz(func: %p, stream: %p, grid: [%u, %u, %u], block: [%u, %u, %u], "
          "shm: %zu, params: %p)", func, stream, gridDim.x, gridDim.y, gridDim.z,
          blockDim.x, blockDim.y, blockDim.z, sharedMem, args);
    return SubmitKernel(func, gridDim, blockDim, args, sharedMem, stream, true);
}

cudaError_t XRtLaunchKernelExC(const cudaLaunchConfig_t *config, const void *func, void **args)
{
    XDEBG("XRtLaunchKernelExC(func: %p, stream: %p, grid: [%u, %u, %u], block: [%u, %u, %u], "
          "shm: %zu, params: %p)", func, config->stream,
          config->gridDim.x, config->gridDim.y, config->gridDim.z,
          config->blockDim.x, config->blockDim.y, config->blockDim.z,
          config->dynamicSmemBytes, args);
    if (config == nullptr) return RtDriver::LaunchKernelExC(config, func, args);
    return SubmitKernel(func, config->gridDim, config->blockDim, args, config->dynamicSmemBytes, config->stream, false);
}

cudaError_t XRtFree(void *devPtr)
{
    XQueueManager::ForEachWaitAll();
    return RtDriver::Free(devPtr);
}

cudaError_t XRtEventQuery(cudaEvent_t event)
{
    XDEBG("XRtEventQuery(event: %p)", event);
    if (event == nullptr) return RtDriver::EventQuery(event);
    auto xevent = g_events.Get(event, nullptr);
    if (xevent == nullptr) return RtDriver::EventQuery(event);

    auto state = xevent->GetState();
    if (state >= kCommandStateCompleted) return cudaSuccess;
    return cudaErrorNotReady;
}

static cudaError_t SubmitEventRecord(cudaEvent_t event, cudaStream_t stream, bool ptsz)
{
    if (event == nullptr) return COREXRT_LAUNCH(ptsz, EventRecord, event, stream);

    cudaError_t result;
    auto xevent = std::make_shared<CorexEventRecordCommand>(event, ptsz);
    if (stream == nullptr || stream == cudaStreamLegacy) {
        WaitBlockingXQueues();
        result = COREXRT_LAUNCH(ptsz, EventRecord, event, stream);
    } else {
        auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
        if (xq == nullptr) {
            result = COREXRT_LAUNCH(ptsz, EventRecord, event, stream);
        } else {
            xq->Submit(xevent);
            result = cudaSuccess;
        }
    }

    g_events.Add(event, xevent);
    return result;
}

cudaError_t XRtEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    XDEBG("XRtEventRecord(event: %p, stream: %p)", event, stream);
    return SubmitEventRecord(event, stream, false);
}

cudaError_t XRtEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream)
{
    XDEBG("XRtEventRecord_ptsz(event: %p, stream: %p)", event, stream);
    return SubmitEventRecord(event, stream, true);
}

cudaError_t XRtEventSynchronize(cudaEvent_t event)
{
    XDEBG("XRtEventSynchronize(event: %p)", event);
    if (event == nullptr) return RtDriver::EventSynchronize(event);

    auto xevent = g_events.Get(event, nullptr);
    if (xevent == nullptr) return RtDriver::EventSynchronize(event);

    xevent->Wait();
    return cudaSuccess;
}

static cudaError_t SubmitEventWait(cudaStream_t stream, cudaEvent_t event, unsigned int flags, bool ptsz)
{
    if (event == nullptr) return COREXRT_LAUNCH(ptsz, StreamWaitEvent, stream, event, flags);
    auto xevent = g_events.Get(event, nullptr);
    // the event is not recorded yet
    if (xevent == nullptr) return COREXRT_LAUNCH(ptsz, StreamWaitEvent, stream, event, flags);

    if (stream == nullptr || stream == cudaStreamLegacy) {
        // sync a event on default stream
        WaitBlockingXQueues();
        xevent->Wait();
        return COREXRT_LAUNCH(ptsz, StreamWaitEvent, stream, event, flags);
    }

    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) {
        // waiting stream is not an xqueue
        if (xevent->GetXQueueHandle() == 0) {
            // the event is not recorded on an xqueue
            return COREXRT_LAUNCH(ptsz, StreamWaitEvent, stream, event, flags);
        }
        xevent->Wait();
        return cudaSuccess;
    }

    auto cmd = std::make_shared<CorexEventWaitCommand>(xevent, flags, ptsz);
    xq->Submit(cmd);
    return cudaSuccess;
}

cudaError_t XRtStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    XDEBG("XRtStreamWaitEvent(stream: %p, event: %p, flags: %u)", stream, event, flags);
    return SubmitEventWait(stream, event, flags, false);
}

cudaError_t XRtStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    XDEBG("XRtStreamWaitEvent_ptsz(stream: %p, event: %p, flags: %u)", stream, event, flags);
    return SubmitEventWait(stream, event, flags, true);
}

cudaError_t XRtEventDestroy(cudaEvent_t event)
{
    XDEBG("XRtEventDestroy(event: %p)", event);
    if (event == nullptr) return RtDriver::EventDestroy(event);

    auto xevent = g_events.DoThenDel(event, nullptr, [](auto xevent) {
        // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef
        // According to CUDA driver API documentation, if the event is waiting in XQueues,
        // we should not destroy it immediately. Instead, we shall set a flag to destroy
        // the CUevent in the destructor of the xevent.
        xevent->DestroyEvent();
    });
    if (xevent == nullptr) return RtDriver::EventDestroy(event);
    return cudaSuccess;
}

cudaError_t XRtDeviceSynchronize()
{
    XDEBG("XRtDeviceSynchronize()");
    XQueueManager::ForEachWaitAll();
    return RtDriver::DeviceSynchronize();
}

cudaError_t XRtStreamSynchronize(cudaStream_t stream)
{
    XDEBG("XRtStreamSynchronize(stream: %p)", stream);
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return RtDriver::StreamSynchronize(stream);
    xq->WaitAll();
    return cudaSuccess;
}

cudaError_t XRtStreamSynchronize_ptsz(cudaStream_t stream)
{
    XDEBG("XRtStreamSynchronize_ptsz(stream: %p)", stream);
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return RtDriver::StreamSynchronize_ptsz(stream);
    xq->WaitAll();
    return cudaSuccess;
}

CUresult XStreamSynchronize(CUstream stream)
{
    XDEBG("XStreamSynchronize(stream: %p)", stream);
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle((cudaStream_t)stream));
    if (xq == nullptr) return Driver::StreamSynchronize(stream);
    xq->WaitAll();
    return CUDA_SUCCESS;
}

CUresult XStreamSynchronize_ptsz(CUstream stream)
{
    XDEBG("XStreamSynchronize_ptsz(stream: %p)", stream);
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle((cudaStream_t)stream));
    if (xq == nullptr) return Driver::StreamSynchronize_ptsz(stream);
    xq->WaitAll();
    return CUDA_SUCCESS;
}

cudaError_t XRtStreamQuery(cudaStream_t stream)
{
    XDEBG("XRtStreamQuery(stream: %p)", stream);
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return RtDriver::StreamQuery(stream);

    switch (xq->Query())
    {
    case kQueueStateIdle:
        return cudaSuccess;
    case kQueueStateReady:
        return cudaErrorNotReady;
    default:
        return RtDriver::StreamQuery(stream);
    }
}

cudaError_t XRtStreamQuery_ptsz(cudaStream_t stream)
{
    XDEBG("XRtStreamQuery_ptsz(stream: %p)", stream);
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return RtDriver::StreamQuery_ptsz(stream);

    switch (xq->Query())
    {
    case kQueueStateIdle:
        return cudaSuccess;
    case kQueueStateReady:
        return cudaErrorNotReady;
    default:
        return RtDriver::StreamQuery_ptsz(stream);
    }
}

cudaError_t XRtStreamCreate(cudaStream_t *stream)
{
    auto res = RtDriver::StreamCreate(stream);
    if (res != cudaSuccess) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return CorexQueueCreate(hwq, *stream); });
    XDEBG("XRtStreamCreate(stream: %p) = %d", *stream, res);
    return res;
}

cudaError_t XRtStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags)
{
    auto res = RtDriver::StreamCreateWithFlags(stream, flags);
    if (res != cudaSuccess) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return CorexQueueCreate(hwq, *stream); });
    XDEBG("XRtStreamCreateWithFlags(stream: %p, flags: 0x%x) = %d", *stream, flags, res);
    return res;
}

cudaError_t XRtStreamCreateWithPriority(cudaStream_t *stream, unsigned int flags, int priority)
{
    auto res = RtDriver::StreamCreateWithPriority(stream, flags, priority);
    if (res != cudaSuccess) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return CorexQueueCreate(hwq, *stream); });
    XDEBG("XRtStreamCreateWithPriority(stream: %p, flags: 0x%x, priority: %d) = %d", *stream, flags, priority, res);
    return res;
}

cudaError_t XRtStreamDestroy(cudaStream_t stream)
{
    XDEBG("XRtStreamDestroy(stream: %p)", stream);
    XQueueManager::AutoDestroy(GetHwQueueHandle(stream));
    return RtDriver::StreamDestroy(stream);
}

CUresult XStreamCreate(CUstream *stream, unsigned int flags)
{
    auto res = Driver::StreamCreate(stream, flags);
    if (res != CUDA_SUCCESS) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return CorexQueueCreate(hwq, (cudaStream_t)*stream); });
    XDEBG("XStreamCreate(stream: %p, flags: 0x%x)", *stream, flags);
    return res;
}

CUresult XStreamCreateWithPriority(CUstream *stream, unsigned int flags, int priority)
{
    auto res = Driver::StreamCreateWithPriority(stream, flags, priority);
    if (res != CUDA_SUCCESS) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return CorexQueueCreate(hwq, (cudaStream_t)*stream); });
    XDEBG("XStreamCreateWithPriority(stream: %p, flags: 0x%x, prio: %d)", *stream, flags, priority);
    return res;
}

CUresult XStreamDestroy_v2(CUstream stream)
{
    XDEBG("XStreamDestroy_v2(stream: %p)", stream);
    XQueueManager::AutoDestroy(GetHwQueueHandle((cudaStream_t)stream));
    return Driver::StreamDestroy_v2(stream);
}

} // namespace xsched::corex
