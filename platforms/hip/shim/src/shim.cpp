#include <list>
#include <mutex>
#include <memory>
#include <unordered_map>

#include "xsched/utils/map.h"
#include "xsched/utils/xassert.h"
#include "xsched/hip/shim/shim.h"
#include "xsched/hip/hal/hal.h"
#include "xsched/hip/hal/handle.h"
#include "xsched/hip/hal/hip_queue.h"
#include "xsched/hip/hal/hip_command.h"
#include "xsched/hip/hal/kernel_param.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/protocol/def.h"
#include "xsched/utils/env.h"

using namespace xsched::preempt;

namespace xsched::hip
{

static std::mutex blocking_xqueue_mutex;
static std::unordered_map<XQueueHandle, std::shared_ptr<XQueue>> blocking_xqueues;
static xsched::utils::ObjectMap<hipEvent_t, std::shared_ptr<HipEventRecordCommand>> g_events;

void HipSyncBlockingXQueues()
{
    std::list<std::shared_ptr<XCommand>> sync_commands;
    blocking_xqueue_mutex.lock();
    for (auto it : blocking_xqueues) sync_commands.emplace_back(it.second->SubmitWaitAll());
    blocking_xqueue_mutex.unlock();
    for (auto sync_command : sync_commands) sync_command->Wait();
}

// Get or create an XQueue for any stream, including the default stream (nullptr).
// This is the equivalent of vllm_xsched's GetXQueueCached — without it, vLLM's
// default-stream kernel launches bypass XSched entirely.
static inline std::shared_ptr<XQueue> GetOrCreateXQueue(hipStream_t stream)
{
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (!xq) {
        XQueueManager::AutoCreate([&](HwQueueHandle *hwq) -> XResult {
            return HipQueueCreate(hwq, stream);
        });
        xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    }
    return xq;
}

hipError_t XLaunchKernel(const void *f, dim3 numBlocks, dim3 dimBlocks, void **args,
                         size_t sharedMemBytes, hipStream_t stream)
{
    if (stream == nullptr) {
        HipSyncBlockingXQueues();
    }
    return Driver::LaunchKernel(f, numBlocks, dimBlocks, args, sharedMemBytes, stream);
}

hipError_t XModuleLaunchKernel(hipFunction_t function,
                              unsigned int gdx, unsigned int gdy, unsigned int gdz,
                              unsigned int bdx, unsigned int bdy, unsigned int bdz,
                              unsigned int shm, hipStream_t stream,
                              void **params, void **extra)
{
    if (stream == nullptr) {
        HipSyncBlockingXQueues();
    }
    return Driver::ModuleLaunchKernel(function, gdx, gdy, gdz, bdx, bdy, bdz, shm,
                                      stream, params, extra);
}

hipError_t XExtModuleLaunchKernel(hipFunction_t f, uint32_t gwx, uint32_t gwy, uint32_t gwz,
                                  uint32_t lwx, uint32_t lwy, uint32_t lwz, size_t shm,
                                  hipStream_t stream, void** params, void** extra,
                                  hipEvent_t start_event, hipEvent_t stop_event, uint32_t flags)
{
    if (stream == nullptr) {
        HipSyncBlockingXQueues();
    }
    return Driver::ExtModuleLaunchKernel(f, gwx, gwy, gwz, lwx, lwy, lwz, shm, stream,
                                         params, extra, start_event, stop_event, flags);
}

void** XRegisterFatBinary(const void* data)
{
    KernelParamManager::Instance()->RegisterStaticCodeObject(data);
    return Driver::RegisterFatBinary(data);
}

void XRegisterFunction(void** modules, const void* hostFunction, char* deviceFunction, const char* deviceName, unsigned int threadLimit, void* tid, void* bid, dim3* blockDim, dim3* gridDim, int* wSize)
{
    XDEBG("XRegisterFunction, hostFunction: %p, deviceName: %s", hostFunction, deviceName);
    KernelParamManager::Instance()->RegisterStaticFunction(hostFunction, deviceName);
    Driver::RegisterFunction(modules, hostFunction, deviceFunction, deviceName, threadLimit, tid, bid, blockDim, gridDim, wSize);
}

hipError_t XMalloc(void **ptr, size_t size)
{
    (void)XCtxSynchronize(); // sync before malloc
    auto res = Driver::Malloc(ptr, size);
    XDEBG("XMalloc %zu bytes at %p, ret: %d", size, ptr ? *ptr : nullptr, res);
    return res;
}

hipError_t XFree(void *ptr)
{
    (void)XCtxSynchronize(); // sync before free
    auto res = Driver::Free(ptr);
    XDEBG("XFree %p, ret: %d", ptr, res);
    return res;
}

hipError_t XMemcpyAsync(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream)
{
    XDEBG("XMemcpyAsync %p -> %p, size: %zu, kind: %d, stream: %p", dst, src, sizeBytes, kind, stream);
    HIP_ASSERT(XStreamSynchronize(stream)); // See also hipMemcpyWithStream
    return Driver::MemcpyAsync(dst, src, sizeBytes, kind, stream);
}

hipError_t XMemcpyWithStream(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream)
{
    XDEBG("XMemcpyWithStream %p -> %p, size: %zu, kind: %d, stream: %p", dst, src, sizeBytes, kind, stream);
    // IMPORTANT: this is a workaround for the unpinned memory issue
    //
    // The user may call hipMemcpyWithStream using unpinned host memory.
    // Without interception, the unpinned memory is copied "synchronously".
    // This is problematic for the XSched, since we really make the memcpy asynchronous.
    //
    // So we manually synchronize the stream here.
    //
    // TODO: we can also check if the memory is pinned, and if so, bypass this synchronization.
    HIP_ASSERT(XStreamSynchronize(stream));
    return Driver::MemcpyWithStream(dst, src, sizeBytes, kind, stream);
}

hipError_t XEventQuery(hipEvent_t event)
{
    if (event == nullptr) return Driver::EventQuery(event);
    auto xevent = g_events.Get(event, nullptr);
    if (xevent == nullptr) return Driver::EventQuery(event);
    auto state = xevent->GetState();
    if (state >= kCommandStateCompleted) return hipSuccess;
    return hipErrorNotReady;
}

hipError_t XEventRecord(hipEvent_t event, hipStream_t stream)
{
    if (event == nullptr) return Driver::EventRecord(event, stream);
    if (stream == nullptr) HipSyncBlockingXQueues();

    // Always dispatch directly to GPU — routing events through XQueue
    // breaks cross-stream event sync that PyTorch/vLLM rely on.
    g_events.Add(event, std::make_shared<HipEventRecordCommand>(event));
    return Driver::EventRecord(event, stream);
}

hipError_t XEventRecordWithFlags(hipEvent_t event, hipStream_t stream, unsigned int flags)
{
    if (event == nullptr) return Driver::EventRecord(event, stream);
    if (stream == nullptr) HipSyncBlockingXQueues();

    g_events.Add(event, std::make_shared<HipEventRecordWithFlagsCommand>(event, flags));
    return Driver::EventRecord(event, stream);
}

hipError_t XEventSynchronize(hipEvent_t event)
{
    if (event == nullptr) return Driver::EventSynchronize(event);
    auto xevent = g_events.Get(event, nullptr);
    if (xevent == nullptr) return Driver::EventSynchronize(event);
    xevent->Wait();
    return hipSuccess;
}

hipError_t XStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags)
{
    if (event == nullptr) return Driver::StreamWaitEvent(stream, event, flags);
    auto xevent = g_events.Get(event, nullptr);
    if (xevent == nullptr) return Driver::StreamWaitEvent(stream, event, flags);

    // Default-stream event wait must be dispatched directly.
    // Routing through XQueue breaks event-based synchronization.
    if (stream == nullptr) {
        HipSyncBlockingXQueues();
        xevent->Synchronize();
        return Driver::StreamWaitEvent(stream, event, flags);
    }

    auto xqueue = GetOrCreateXQueue(stream);
    if (xqueue == nullptr) {
        if (xevent->GetXQueueHandle() == 0) {
            return Driver::StreamWaitEvent(stream, event, flags);
        }
        xevent->Synchronize();
        return hipSuccess;
    }

    auto command = std::make_shared<HipEventWaitCommand>(xevent, flags);
    xqueue->Submit(command);
    return hipSuccess;
}

hipError_t XEventDestroy(hipEvent_t event)
{
    if (event == nullptr) return Driver::EventDestroy(event);
    auto xevent = g_events.DoThenDel(event, nullptr, [](auto xevent) { xevent->DestroyEvent(); });
    if (xevent == nullptr) return Driver::EventDestroy(event);
    // According to HIP driver API documentation, if the event is waiting
    // in XQueues, we should not destroy it immediately. Instead, we shall
    // set a flag to destroy the hipEvent in the destructor of the xevent.
    return hipSuccess;
}

hipError_t XStreamSynchronize(hipStream_t stream)
{
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return Driver::StreamSynchronize(stream);
    xq->WaitAll();
    return hipSuccess;
}

hipError_t XStreamQuery(hipStream_t stream)
{
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return Driver::StreamQuery(stream);
    switch (xq->Query())
    {
    case kQueueStateIdle:
        return hipSuccess;
    case kQueueStateReady:
        return hipErrorNotReady;
    default:
        return Driver::StreamQuery(stream);
    }
}

hipError_t XCtxSynchronize()
{
    XQueueManager::ForEachWaitAll();
    return Driver::CtxSynchronize();
}

hipError_t XStreamCreate(hipStream_t *stream)
{
    int64_t prio = PRIORITY_DEFAULT;
    GetEnvInt64(XSCHED_AUTO_XQUEUE_PRIORITY_ENV_NAME, prio);
    hipError_t res = Driver::StreamCreateWithPriority(stream, 0, (int)prio);
    if (res != hipSuccess) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return HipQueueCreate(hwq, *stream); });
    XDEBG("XStreamCreate(stream: %p, prio: " FMT_64D ")", *stream, prio);
    return res;
}

hipError_t XStreamCreateWithFlags(hipStream_t *stream, unsigned int flags)
{
    int64_t prio = PRIORITY_DEFAULT;
    GetEnvInt64(XSCHED_AUTO_XQUEUE_PRIORITY_ENV_NAME, prio);
    hipError_t res = Driver::StreamCreateWithPriority(stream, flags, (int)prio);
    if (res != hipSuccess) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return HipQueueCreate(hwq, *stream); });
    XDEBG("XStreamCreateWithFlags(stream: %p, flags: 0x%x, prio: " FMT_64D ")", *stream, flags, prio);
    return res;
}

hipError_t XStreamCreateWithPriority(hipStream_t *stream, unsigned int flags, int priority)
{
    hipError_t res = Driver::StreamCreateWithPriority(stream, flags, priority);
    if (res != hipSuccess) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return HipQueueCreate(hwq, *stream); });
    XDEBG("XStreamCreateWithPriority(stream: %p, flags: 0x%x, priority: %d)", *stream, flags, priority);
    return res;
}

hipError_t XStreamDestroy(hipStream_t stream)
{
    XDEBG("XStreamDestroy(stream: %p)", stream);
    XQueueManager::AutoDestroy(GetHwQueueHandle(stream));
    return Driver::StreamDestroy(stream);
}

} // namespace xsched::shim::hip
