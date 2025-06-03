#include <list>

#include "xsched/xqueue.h"
#include "xsched/utils/map.h"
#include "xsched/protocol/def.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/preempt/xqueue/xqueue.h"
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
    std::list<std::shared_ptr<XQueueWaitAllCommand>> wait_cmds;
    XResult res = XQueueManager::ForEach([&](std::shared_ptr<XQueue> xq)->XResult {
        auto hwq = xq->GetHwQueue();
        auto cuda_q = std::dynamic_pointer_cast<CudaQueueL1>(hwq);
        if (cuda_q == nullptr) return kXSchedErrorUnknown;
        // does not need to wait a non-blocking stream
        if (cuda_q->GetStreamFlags() & CU_STREAM_NON_BLOCKING) return kXSchedSuccess;
        auto wait_cmd = xq->SubmitWaitAll();
        if (wait_cmd == nullptr) return kXSchedErrorUnknown;
        wait_cmds.push_back(wait_cmd);
        return kXSchedSuccess;
    });
    XASSERT(res == kXSchedSuccess, "Fail to submit wait all commands");
    for (auto &cmd : wait_cmds) cmd->Wait();
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
        auto kernel = std::make_shared<CudaKernelLaunchCommand>(
            f, gdx, gdy, gdz, bdx, bdy, bdz, shmem, params, extra, false);
        return DirectLaunch(kernel, stream);
    }

    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    auto kernel = std::make_shared<CudaKernelLaunchCommand>(
        f, gdx, gdy, gdz, bdx, bdy, bdz, shmem, params, extra, xq != nullptr);

    if (xq == nullptr) return DirectLaunch(kernel, stream);
    xq->Submit(kernel);
    return CUDA_SUCCESS;
}

CUresult XLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **params, void **extra)
{
    XDEBG("XLaunchKernelEx(cfg: %p, func: %p, params: %p, extra: %p)", config, f, params, extra);
    if (config == nullptr) return Driver::LaunchKernelEx(config, f, params, extra);

    CUstream stream = config->hStream;

    if (stream == nullptr) {
        WaitBlockingXQueues();
        auto kernel = std::make_shared<CudaKernelLaunchExCommand>(config, f, params, extra, false);
        return DirectLaunch(kernel, stream);
    }
    
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    auto kn = std::make_shared<CudaKernelLaunchExCommand>(config, f, params, extra, xq != nullptr);

    if (xq == nullptr) return DirectLaunch(kn, stream);
    xq->Submit(kn);
    return CUDA_SUCCESS;
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

    CUresult result;
    auto xevent = std::make_shared<CudaEventRecordCommand>(event);

    if (stream == nullptr) {
        WaitBlockingXQueues();
        result = Driver::EventRecord(event, stream);
    } else {
        auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
        if (xq == nullptr) {
            result = Driver::EventRecord(event, stream);
        } else {
            xq->Submit(xevent);
            result = CUDA_SUCCESS;
        }
    }

    g_events.Add(event, xevent);
    return result;
}

CUresult XEventRecordWithFlags(CUevent event, CUstream stream, unsigned int flags)
{
    XDEBG("XEventRecordWithFlags(event: %p, stream: %p, flags: %u)", event, stream, flags);
    if (event == nullptr) return Driver::EventRecordWithFlags(event, stream, flags);

    CUresult result;
    auto xevent = std::make_shared<CudaEventRecordWithFlagsCommand>(event, flags);

    if (stream == nullptr) {
        WaitBlockingXQueues();
        result = Driver::EventRecordWithFlags(event, stream, flags);
    } else {
        auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
        if (xq == nullptr) {
            result = Driver::EventRecordWithFlags(event, stream, flags);
        } else {
            xq->Submit(xevent);
            result = CUDA_SUCCESS;
        }
    }

    g_events.Add(event, xevent);
    return result;
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
        // sync a event on default stream
        WaitBlockingXQueues();
        xevent->Wait();
        return Driver::StreamWaitEvent(stream, event, flags);
    }

    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) {
        // waiting stream is not an xqueue
        if (xevent->GetXQueueHandle() == 0) {
            // the event is not recorded on an xqueue
            return Driver::StreamWaitEvent(stream, event, flags);
        }
        xevent->Wait();
        return CUDA_SUCCESS;
    }

    auto cmd = std::make_shared<CudaEventWaitCommand>(xevent, flags);
    xq->Submit(cmd);
    return CUDA_SUCCESS;
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

static bool SingleStream()
{
    static char *env = std::getenv("SINGLE_STREAM");
    if (env == nullptr || strlen(env) == 0 || strcmp(env, "0") == 0 ||
        strcasecmp(env, "off") == 0) {
        XDEBG("Single stream is disabled");
        return false;
    }
    return true;
}

CUstream single_stream = nullptr;

CUresult XStreamCreate(CUstream *stream, unsigned int flags)
{
    if (SingleStream() && single_stream != nullptr) {
        *stream = single_stream;
        return CUDA_SUCCESS;
    }

    CUresult res = Driver::StreamCreate(stream, flags);
    if (res != CUDA_SUCCESS) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return CudaQueueCreate(hwq, *stream); });
    XDEBG("XStreamCreate(stream: %p, flags: 0x%x)", *stream, flags);
    single_stream = *stream;
    return res;
}

CUresult XStreamCreateWithPriority(CUstream *stream, unsigned int flags, int priority)
{
    if (SingleStream() && single_stream != nullptr) {
        *stream = single_stream;
        return CUDA_SUCCESS;
    }

    CUresult res = Driver::StreamCreateWithPriority(stream, flags, priority);
    if (res != CUDA_SUCCESS) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return CudaQueueCreate(hwq, *stream); });
    XDEBG("XStreamCreateWithPriority(stream: %p, flags: 0x%x, priority: %d)",
          *stream, flags, priority);
    single_stream = *stream;
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
