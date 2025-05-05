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

hipError_t XLaunchKernel(const void *f, dim3 numBlocks, dim3 dimBlocks, void **args,
                         size_t sharedMemBytes, hipStream_t stream)
{
    if (stream == nullptr) {
        HipSyncBlockingXQueues();
        return Driver::LaunchKernel(f, numBlocks, dimBlocks, args, sharedMemBytes, stream);
    }
    
    auto xqueue = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xqueue == nullptr) {
        return Driver::LaunchKernel(f, numBlocks, dimBlocks, args, sharedMemBytes, stream);
    }
    
    auto kernel = std::make_shared<HipKernelLaunchCommand>(
        f, numBlocks, dimBlocks, args, sharedMemBytes, xqueue != nullptr);
    xqueue->Submit(kernel);
    return hipSuccess;
}

hipError_t XModuleLaunchKernel(hipFunction_t function,
                              unsigned int gdx, unsigned int gdy, unsigned int gdz,
                              unsigned int bdx, unsigned int bdy, unsigned int bdz,
                              unsigned int shm, hipStream_t stream,
                              void **params, void **extra)
{
    if (stream == nullptr) {
        HipSyncBlockingXQueues();
        return Driver::ModuleLaunchKernel(function, gdx, gdy, gdz, bdx, bdy, bdz, shm,
                                          stream, params, extra);
    }

    auto xqueue = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xqueue == nullptr) {
        return Driver::ModuleLaunchKernel(function, gdx, gdy, gdz, bdx, bdy, bdz, shm,
                                          stream, params, extra);
    }

    auto kernel = std::make_shared<HipModuleKernelLaunchCommand>(function, gdx, gdy, gdz, bdx, bdy, bdz, shm,
                                                                 params, extra, xqueue != nullptr);
    xqueue->Submit(kernel);
    return hipSuccess;
}

hipError_t XExtModuleLaunchKernel(hipFunction_t f, uint32_t gwx, uint32_t gwy, uint32_t gwz,
                                  uint32_t lwx, uint32_t lwy, uint32_t lwz, size_t shm,
                                  hipStream_t stream, void** params, void** extra,
                                  hipEvent_t start_event, hipEvent_t stop_event, uint32_t flags)
{
    if (stream == nullptr) {
        HipSyncBlockingXQueues();
        return Driver::ExtModuleLaunchKernel(f, gwx, gwy, gwz, lwx, lwy, lwz, shm, stream,
                                             params, extra, start_event, stop_event, flags);
    }

    auto xqueue = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xqueue == nullptr) {
        return Driver::ExtModuleLaunchKernel(f, gwx, gwy, gwz, lwx, lwy, lwz, shm, stream,
                                             params, extra, start_event, stop_event, flags);
    }

    auto kernel = std::make_shared<HipExtModuleKernelLaunchCommand>(
        f, gwx, gwy, gwz, lwx, lwy, lwz, shm, params, extra, start_event, stop_event, flags, xqueue != nullptr);
    xqueue->Submit(kernel);
    return hipSuccess;
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
    XCtxSynchronize(); // sync before malloc
    auto res = Driver::Malloc(ptr, size);
    XDEBG("XMalloc %zu bytes at %p, ret: %d", size, ptr ? *ptr : nullptr, res);
    return res;
}

hipError_t XFree(void *ptr)
{
    XCtxSynchronize(); // sync before free
    auto res = Driver::Free(ptr);
    XDEBG("XFree %p, ret: %d", ptr, res);
    return res;
}

hipError_t XMemcpyAsync(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream)
{
    XDEBG("XMemcpyAsync %p -> %p, size: %zu, kind: %d, stream: %p", dst, src, sizeBytes, kind, stream);
    XStreamSynchronize(stream); // See also hipMemcpyWithStream
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
    XStreamSynchronize(stream);
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

    hipError_t result;
    auto command = std::make_shared<HipEventRecordCommand>(event);

    if (stream == nullptr) {
        HipSyncBlockingXQueues();
        result = Driver::EventRecord(event, stream);
    } else {
        auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
        if (xq == nullptr) {
            result = Driver::EventRecord(event, stream);
        } else {
            xq->Submit(command);
            result = hipSuccess;
        }
    }

    g_events.Add(event, command);
    return result;
}

hipError_t XEventRecordWithFlags(hipEvent_t event, hipStream_t stream, unsigned int flags)
{
    if (event == nullptr) return Driver::EventRecord(event, stream);

    hipError_t result;
    auto command = std::make_shared<HipEventRecordWithFlagsCommand>(event, flags);

    if (stream == nullptr) {
        HipSyncBlockingXQueues();
        result = Driver::EventRecord(event, stream);
    } else {
        auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
        if (xq == nullptr) {
            result = Driver::EventRecord(event, stream);
        } else {
            xq->Submit(command);
            result = hipSuccess;
        }
    }

    g_events.Add(event, command);
    return result;
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

    if (stream == nullptr) {
        // sync a event on default stream
        HipSyncBlockingXQueues();
        xevent->Synchronize();
        return Driver::StreamWaitEvent(stream, event, flags);
    }

    auto xqueue = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xqueue == nullptr) {
        // waiting stream is not a xqueue
        if (xevent->GetXQueueHandle() == 0) {
            // the event is not recorded on a xqueue
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
    hipError_t res = Driver::StreamCreate(stream);
    if (res != hipSuccess) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return HipQueueCreate(hwq, *stream); });
    XDEBG("XStreamCreate(stream: %p)", *stream);
    return res;
}

hipError_t XStreamCreateWithFlags(hipStream_t *stream, unsigned int flags)
{
    hipError_t res = Driver::StreamCreateWithFlags(stream, flags);
    if (res != hipSuccess) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return HipQueueCreate(hwq, *stream); });
    XDEBG("XStreamCreateWithFlags(stream: %p, flags: 0x%x)", *stream, flags);
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
