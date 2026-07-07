#include <list>
#include <thread>
#include <chrono>

#include "xsched/xqueue.h"
#include "xsched/utils/map.h"
#include "xsched/protocol/def.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/preempt/sched/agent.h"
#include "xsched/cuda/hal.h"
#include "xsched/cuda/shim/shim.h"
#include "xsched/cuda/hal/common/levels.h"
#include "xsched/cuda/hal/level1/cuda_queue.h"
#include "xsched/cuda/hal/common/cuda_command.h"

using namespace xsched::preempt;

namespace xsched::cuda
{

static utils::ObjectMap<CUevent, std::shared_ptr<CudaEventRecordCommand>> g_events;

void WaitBlockingXQueues()
{
    // Empty — same pattern as HIP.
}

// Get or create an XQueue for any stream, including the default stream.
static inline std::shared_ptr<XQueue> GetOrCreateXQueue(CUstream stream)
{
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (!xq) {
        XQueueManager::AutoCreate([&](HwQueueHandle *hwq) -> XResult {
            return CudaQueueCreate(hwq, stream);
        });
        xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    }
    return xq;
}

CUresult XLaunchKernel(CUfunction f,
                       unsigned int gdx, unsigned int gdy, unsigned int gdz,
                       unsigned int bdx, unsigned int bdy, unsigned int bdz,
                       unsigned int shmem, CUstream stream, void **params, void **extra)
{
    XDEBG("XLaunchKernel(func: %p, stream: %p, grid: [%u, %u, %u], block: [%u, %u, %u], "
          "shm: %u, params: %p, extra: %p)", f, stream, gdx, gdy, gdz, bdx, bdy, bdz,
          shmem, params, extra);

    if (stream == nullptr) {
        WaitBlockingXQueues();
    }

    auto xq = GetOrCreateXQueue(stream);

    // Ready heartbeat
    if (xq) {
        static thread_local auto last_ready = std::chrono::steady_clock::time_point();
        auto now = std::chrono::steady_clock::now();
        if (now - last_ready > std::chrono::seconds(1)) {
            xsched::preempt::SchedAgent::SendEvent(
                std::make_shared<xsched::sched::XQueueReadyEvent>(
                    xq->GetHandle(), std::chrono::system_clock::now()));
            last_ready = now;
        }
    }

    // Suspend gate
    if (xq && xq->IsSuspended()) {
        while (xq->IsSuspended()) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    auto kernel = std::make_shared<CudaKernelLaunchCommand>(
        f, gdx, gdy, gdz, bdx, bdy, bdz, shmem, params, extra, xq != nullptr);
    return DirectLaunch(kernel, stream);
}

CUresult XLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **params, void **extra)
{
    XDEBG("XLaunchKernelEx(cfg: %p, func: %p, params: %p, extra: %p)", config, f, params, extra);
    if (config == nullptr) return Driver::LaunchKernelEx(config, f, params, extra);

    CUstream stream = config->hStream;

    if (stream == nullptr) {
        WaitBlockingXQueues();
    }

    auto xq = GetOrCreateXQueue(stream);

    if (xq) {
        static thread_local auto last_ready = std::chrono::steady_clock::time_point();
        auto now = std::chrono::steady_clock::now();
        if (now - last_ready > std::chrono::seconds(1)) {
            xsched::preempt::SchedAgent::SendEvent(
                std::make_shared<xsched::sched::XQueueReadyEvent>(
                    xq->GetHandle(), std::chrono::system_clock::now()));
            last_ready = now;
        }
        if (xq->IsSuspended()) {
            while (xq->IsSuspended()) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
    }

    auto kn = std::make_shared<CudaKernelLaunchExCommand>(config, f, params, extra, xq != nullptr);
    return DirectLaunch(kn, stream);
}

CUresult XLaunchHostFunc(CUstream stream, CUhostFn fn, void *data)
{
    if (stream == 0) {
        WaitBlockingXQueues();
        return Driver::LaunchHostFunc(stream, fn, data);
    }
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return Driver::LaunchHostFunc(stream, fn, data);
    auto hw_cmd = std::make_shared<CudaHostFuncCommand>(fn, data);
    xq->Submit(hw_cmd);
    return CUDA_SUCCESS;
}

CUresult XEventQuery(CUevent event)
{
    XDEBG("XEventQuery(event: %p)", event);
    if (event == nullptr) return Driver::EventQuery(event);
    auto xevent = g_events.Get(event, nullptr);
    if (xevent == nullptr) return Driver::EventQuery(event);

    auto state = xevent->GetState();
    if (state >= kCommandStateCompleted) return CUDA_SUCCESS;
    return CUDA_ERROR_NOT_READY;
}

CUresult XEventRecord(CUevent event, CUstream stream)
{
    XDEBG("XEventRecord(event: %p, stream: %p)", event, stream);
    if (event == nullptr) return Driver::EventRecord(event, stream);
    if (stream == nullptr) WaitBlockingXQueues();

    // Always dispatch directly — routing events through XQueue breaks
    // cross-stream event sync (same bug as HIP, causes illegal memory access).
    g_events.Add(event, std::make_shared<CudaEventRecordCommand>(event));
    return Driver::EventRecord(event, stream);
}

CUresult XEventRecordWithFlags(CUevent event, CUstream stream, unsigned int flags)
{
    XDEBG("XEventRecordWithFlags(event: %p, stream: %p, flags: %u)", event, stream, flags);
    if (event == nullptr) return Driver::EventRecordWithFlags(event, stream, flags);
    if (stream == nullptr) WaitBlockingXQueues();

    g_events.Add(event, std::make_shared<CudaEventRecordWithFlagsCommand>(event, flags));
    return Driver::EventRecord(event, stream);
}

CUresult XEventSynchronize(CUevent event)
{
    XDEBG("XEventSynchronize(event: %p)", event);
    if (event == nullptr) return Driver::EventSynchronize(event);

    auto xevent = g_events.Get(event, nullptr);
    if (xevent == nullptr) return Driver::EventSynchronize(event);

    xevent->Wait();
    return CUDA_SUCCESS;
}

CUresult XStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags)
{
    XDEBG("XStreamWaitEvent(stream: %p, event: %p, flags: %u)", stream, event, flags);
    if (event == nullptr)return Driver::StreamWaitEvent(stream, event, flags);

    auto xevent = g_events.Get(event, nullptr);
    // the event is not recorded yet
    if (xevent == nullptr) return Driver::StreamWaitEvent(stream, event, flags);

    if (stream == nullptr) {
        WaitBlockingXQueues();
        xevent->Wait();
        return Driver::StreamWaitEvent(stream, event, flags);
    }

    xevent->Wait();
    return Driver::StreamWaitEvent(stream, event, flags);
}

CUresult XEventDestroy(CUevent event)
{
    XDEBG("XEventDestroy(event: %p)", event);
    if (event == nullptr) return Driver::EventDestroy(event);

    auto xevent = g_events.DoThenDel(event, nullptr, [](auto xevent) {
        // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef
        // According to CUDA driver API documentation, if the event is waiting in XQueues,
        // we should not destroy it immediately. Instead, we shall set a flag to destroy
        // the CUevent in the destructor of the xevent.
        xevent->DestroyEvent();
    });
    if (xevent == nullptr) return Driver::EventDestroy(event);
    return CUDA_SUCCESS;
}

CUresult XEventDestroy_v2(CUevent event)
{
    XDEBG("XEventDestroy_v2(event: %p)", event);
    if (event == nullptr) return Driver::EventDestroy_v2(event);

    auto xevent = g_events.DoThenDel(event, nullptr, [](auto xevent) {
        // Same as XEventDestroy.
        xevent->DestroyEvent();
    });
    if (xevent == nullptr) return Driver::EventDestroy_v2(event);
    return CUDA_SUCCESS;
}

CUresult XStreamSynchronize(CUstream stream)
{
    XDEBG("XStreamSynchronize(stream: %p)", stream);
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return Driver::StreamSynchronize(stream);
    xq->WaitAll();
    return CUDA_SUCCESS;
}

CUresult XStreamQuery(CUstream stream)
{
    XDEBG("XStreamQuery(stream: %p)", stream);
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) Driver::StreamQuery(stream);

    switch (xq->Query())
    {
    case kQueueStateIdle:
        return CUDA_SUCCESS;
    case kQueueStateReady:
        return CUDA_ERROR_NOT_READY;
    default:
        return Driver::StreamQuery(stream);
    }
}
CUresult XCtxSynchronize()
{
    XDEBG("XCtxSynchronize()");
    XQueueManager::ForEachWaitAll();
    return Driver::CtxSynchronize();
}

CUresult XStreamCreate(CUstream *stream, unsigned int flags)
{
    CUresult res = Driver::StreamCreate(stream, flags);
    if (res != CUDA_SUCCESS) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return CudaQueueCreate(hwq, *stream); });
    XDEBG("XStreamCreate(stream: %p, flags: 0x%x)", *stream, flags);
    return res;
}

CUresult XStreamCreateWithPriority(CUstream *stream, unsigned int flags, int priority)
{
    CUresult res = Driver::StreamCreateWithPriority(stream, flags, priority);
    if (res != CUDA_SUCCESS) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return CudaQueueCreate(hwq, *stream); });
    XDEBG("XStreamCreateWithPriority(stream: %p, flags: 0x%x, priority: %d)",
          *stream, flags, priority);
    return res;
}

CUresult XStreamDestroy(CUstream stream)
{
    XDEBG("XStreamDestroy(stream: %p)", stream);
    XQueueManager::AutoDestroy(GetHwQueueHandle(stream));
    return Driver::StreamDestroy(stream);
}

CUresult XStreamDestroy_v2(CUstream stream)
{
    XDEBG("XStreamDestroy_v2(stream: %p)", stream);
    XQueueManager::AutoDestroy(GetHwQueueHandle(stream));
    return Driver::StreamDestroy_v2(stream);
}

} // namespace xsched::cuda
