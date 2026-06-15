#include <list>
#include <mutex>
#include <atomic>
#include <functional>

#include "xsched/utils/log.h"
#include "xsched/utils/map.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/cuda/hal.h"
#include "xsched/cuda/shim/shim.h"
#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/common/levels.h"
#include "xsched/cuda/hal/common/options.h"
#include "xsched/cuda/hal/level1/cuda_queue.h"
#include "xsched/cuda/hal/common/cuda_command.h"

using namespace xsched::preempt;

namespace xsched::cuda
{

static utils::ObjectMap<CUevent, std::shared_ptr<CudaEventRecordCommand>> g_events;

CUstream GetPTDS()
{
    /// FIXME: Here we assume that the thread will only use one single CUDA context.
    /// However, each CUDA context should have its own per-thread default stream.
    /// TODO: Destory the stream when the thread exits.
    static thread_local CUstream per_thread_default_stream = 0;
    if (per_thread_default_stream != 0) return per_thread_default_stream;

    CUstream stream = nullptr;
    CUDA_ASSERT(Driver::StreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    /// FIXME: If XSCHED_AUTO_XQUEUE is not turned on,
    /// there is no meaning creating new per-thread default streams.
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) {return CudaQueueCreate(hwq, stream);});
    per_thread_default_stream = stream;
    return stream;
}

void WaitBlockingXQueues()
{
    std::list<std::shared_ptr<XQueueWaitAllCommand>> wait_cmds;
    XResult res = XQueueManager::ForEach([&](std::shared_ptr<XQueue> xq)->XResult {
        auto hwq = xq->GetHwQueue();
        auto cuda_q = std::dynamic_pointer_cast<CudaQueueLv1>(hwq);
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

static std::mutex g_capture_mutex;
static std::atomic<int64_t> g_capture_counter {0};

void CaptureBegin()
{
    /// @note: We assume that when a thread starts a CUDA graph capture,
    /// no other threads will submit commands concurrently.
    /// Explanation:
    /// While GetCaptureCounter() > 0, every shim entry point bypasses the
    /// XQueue and launch it command directly to the driver (see CHECK_STREAM*
    /// in shim.h). CaptureBegin / CaptureEnd maintain that counter around
    /// each cuStreamBeginCapture / cuStreamEndCapture pair.
    ///
    /// Two invariants we want to hold for any thread that observes
    /// GetCaptureCounter() > 0:
    ///
    ///   (1) Capture safety. When the driver actually starts capturing a
    ///       stream, no XQueue launch worker is still about to launch
    ///       any commands onto that stream or do synchronization
    ///       — otherwise the command would be silently recorded into the graph
    ///       or cause CUDA graph invalidation if synchronization is called.
    ///
    ///   (2) Ordering safety. Any thread that observes counter > 0 and
    ///       therefore switches to "direct launch mode" must have its
    ///       previously-submitted XQueue commands already completed.
    ///       Otherwise its direct launches can race ahead of commands it
    ///       submitted earlier on the same XQueue.
    ///
    /// Here, we do XCtxSynchronize first, then increase counter.
    /// So that (2) can be guaranteed even when there is a concurrent
    /// command submission.
    /// Corner case: two concurrent threads are submitting to the same XQueue
    /// thread A check count == 0 => thread X ctx sync => thread X count = 1
    /// => thread B check count == 1 => thread A submit command A to XQueue A
    /// => thread B launch command B directly to CUDA => XQueue A launch command A
    /// in this case command A will be executed after command B.
    /// However, in CUDA semantics, command A and B can also be reordered.
    /// But (1) still depends on the application-level assumption that
    /// no other thread submits to an XQueue while CaptureBegin is running.
    std::lock_guard<std::mutex> lock(g_capture_mutex);
    if (g_capture_counter.load() == 0) XCtxSynchronize();
    g_capture_counter.fetch_add(1);
}

void CaptureEnd()
{
    std::lock_guard<std::mutex> lock(g_capture_mutex);
    int64_t counter = g_capture_counter.load();
    if (counter == 0) {
        XWARN("CaptureEnd() called but capture counter is 0");
        return;
    }
    g_capture_counter.store(counter - 1);
}

int64_t GetCaptureCounter()
{
    return g_capture_counter.load();
}

template <typename CmdT, typename... Args>
CUresult XLaunchKernelImpl(CUstream stream, Args&&... args)
{
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    auto kernel = std::make_shared<CmdT>(std::forward<Args>(args)..., xq != nullptr);
    if (xq == nullptr) return DirectLaunch(kernel, stream);
    xq->Submit(kernel);
    return CUDA_SUCCESS;
}

CUresult XLaunchKernel(CUfunction f,
                       unsigned int gdx, unsigned int gdy, unsigned int gdz,
                       unsigned int bdx, unsigned int bdy, unsigned int bdz,
                       unsigned int shmem, CUstream stream, void **params, void **extra)
{
    XDEBG("XLaunchKernel(func: %p, stream: %p, grid: [%u, %u, %u], block: [%u, %u, %u], "
          "shm: %u, params: %p, extra: %p)", f, stream, gdx, gdy, gdz, bdx, bdy, bdz,
          shmem, params, extra);
    CHECK_STREAM(stream, DirectLaunch(std::make_shared<CudaKernelLaunchCommand>(
        f, gdx, gdy, gdz, bdx, bdy, bdz, shmem, params, extra, false), stream));
    return XLaunchKernelImpl<CudaKernelLaunchCommand>(
        stream, f, gdx, gdy, gdz, bdx, bdy, bdz, shmem, params, extra);
}

CUresult XLaunchKernel_ptsz(CUfunction f,
                            unsigned int gdx, unsigned int gdy, unsigned int gdz,
                            unsigned int bdx, unsigned int bdy, unsigned int bdz,
                            unsigned int shmem, CUstream stream, void **params, void **extra)
{
    XDEBG("XLaunchKernel_ptsz(func: %p, stream: %p, grid: [%u, %u, %u], block: [%u, %u, %u], "
          "shm: %u, params: %p, extra: %p)", f, stream, gdx, gdy, gdz, bdx, bdy, bdz,
          shmem, params, extra);
    CHECK_STREAM_PTSZ(stream, DirectLaunch(std::make_shared<CudaKernelLaunchCommand>(
        f, gdx, gdy, gdz, bdx, bdy, bdz, shmem, params, extra, false), stream));
    return XLaunchKernelImpl<CudaKernelLaunchCommand>(
        stream, f, gdx, gdy, gdz, bdx, bdy, bdz, shmem, params, extra);
}

CUresult XLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **params, void **extra)
{
    if (config == nullptr) {
        XDEBG("XLaunchKernelEx(cfg: %p, func: %p, params: %p, extra: %p)",
              config, f, params, extra);
        return Driver::LaunchKernelEx(config, f, params, extra);
    }
    XDEBG("XLaunchKernelEx(cfg: %p, func: %p, params: %p, extra: %p, stream: %p)",
          config, f, params, extra, config->hStream);

    CUlaunchConfig cfg = *config;
    CHECK_STREAM(cfg.hStream, DirectLaunch(std::make_shared<CudaKernelLaunchExCommand>(
        &cfg, f, params, extra, false), cfg.hStream));
    return XLaunchKernelImpl<CudaKernelLaunchExCommand>(cfg.hStream, &cfg, f, params, extra);
}

CUresult XLaunchKernelEx_ptsz(const CUlaunchConfig *config, CUfunction f, void **params, void **extra)
{
    if (config == nullptr) {
        XDEBG("XLaunchKernelEx_ptsz(cfg: %p, func: %p, params: %p, extra: %p)",
              config, f, params, extra);
        return Driver::LaunchKernelEx(config, f, params, extra);
    }
    XDEBG("XLaunchKernelEx_ptsz(cfg: %p, func: %p, params: %p, extra: %p, stream: %p)",
          config, f, params, extra, config->hStream);

    CUlaunchConfig cfg = *config;
    CHECK_STREAM_PTSZ(cfg.hStream, DirectLaunch(std::make_shared<CudaKernelLaunchExCommand>(
        &cfg, f, params, extra, false), cfg.hStream));
    return XLaunchKernelImpl<CudaKernelLaunchExCommand>(cfg.hStream, &cfg, f, params, extra);
}

static inline CUresult XLaunchHostFuncImpl(CUstream stream, CUhostFn fn, void *data)
{
    // stream is the first arg, different from other XLaunch* functions.
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return Driver::LaunchHostFunc(stream, fn, data);
    auto hw_cmd = std::make_shared<CudaHostFuncCommand>(fn, data);
    xq->Submit(hw_cmd);
    return CUDA_SUCCESS;
}

CUresult XLaunchHostFunc(CUstream stream, CUhostFn fn, void *data)
{
    XDEBG("XLaunchHostFunc(stream: %p, fn: %p, data: %p)", stream, fn, data);
    CHECK_STREAM(stream, Driver::LaunchHostFunc(stream, fn, data));
    return XLaunchHostFuncImpl(stream, fn, data);
}

CUresult XLaunchHostFunc_ptsz(CUstream stream, CUhostFn fn, void *data)
{
    XDEBG("XLaunchHostFunc_ptsz(stream: %p, fn: %p, data: %p)", stream, fn, data);
    CHECK_STREAM_PTSZ(stream, Driver::LaunchHostFunc(stream, fn, data));
    return XLaunchHostFuncImpl(stream, fn, data);
}

static inline CUresult XStreamEndCaptureImpl(CUstream stream, CUgraph *graph)
{
    CUstreamCaptureStatus before = CU_STREAM_CAPTURE_STATUS_NONE;
    CUresult qres = Driver::StreamIsCapturing(stream, &before);
    CUresult res = Driver::StreamEndCapture(stream, graph);
    if (res != CUDA_ERROR_STREAM_CAPTURE_UNMATCHED && // end capture on the correct stream
        qres == CUDA_SUCCESS && before != CU_STREAM_CAPTURE_STATUS_NONE) {
        // The stream is capturing before, check if it is actually stopped.
        CUstreamCaptureStatus after = CU_STREAM_CAPTURE_STATUS_NONE;
        qres = Driver::StreamIsCapturing(stream, &after);
        if (qres == CUDA_SUCCESS && after == CU_STREAM_CAPTURE_STATUS_NONE) {
            CaptureEnd();
        }
    }
    return res;
}

CUresult XStreamEndCapture(CUstream stream, CUgraph *graph)
{
    XDEBG("XStreamEndCapture(stream: %p, graph: %p)", stream, graph);
    CONVERT_STREAM(stream);
    return XStreamEndCaptureImpl(stream, graph);
}

CUresult XStreamEndCapture_ptsz(CUstream stream, CUgraph *graph)
{
    XDEBG("XStreamEndCapture_ptsz(stream: %p, graph: %p)", stream, graph);
    CONVERT_STREAM_PTSZ(stream);
    return XStreamEndCaptureImpl(stream, graph);
}

CUresult XMemFree_v2(CUdeviceptr dptr)
{
    /// TODO: Optimize this.
    /// In CUDA semantics, cuMemFree only waits for commands who use this memory.
    XQueueManager::ForEachWaitAll();
    return Driver::MemFree_v2(dptr);
}

static CUresult XEventRecordImpl(std::shared_ptr<CudaEventRecordCommand> xevent,
                                 CUevent event, CUstream stream)
{
    CUresult result;
    if (GetCaptureCounter() > 0) {
        result = xevent->LaunchWrapper(stream);
    } else if (stream == CU_STREAM_LEGACY) {
        WaitBlockingXQueues();
        result = xevent->LaunchWrapper(stream);
    } else {
        auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
        if (xq == nullptr) {
            result = xevent->LaunchWrapper(stream);
        } else {
            xq->Submit(xevent);
            result = CUDA_SUCCESS;
        }
    }
    g_events.Add(event, xevent);
    return result;
}

CUresult XEventRecord(CUevent event, CUstream stream)
{
    XDEBG("XEventRecord(event: %p, stream: %p)", event, stream);
    CONVERT_STREAM(stream);
    if (event == nullptr) return Driver::EventRecord(event, stream);
    auto xevent = std::make_shared<CudaEventRecordCommand>(event);
    return XEventRecordImpl(xevent, event, stream);
}

CUresult XEventRecord_ptsz(CUevent event, CUstream stream)
{
    XDEBG("XEventRecord_ptsz(event: %p, stream: %p)", event, stream);
    CONVERT_STREAM_PTSZ(stream);
    if (event == nullptr) return Driver::EventRecord(event, stream);
    auto xevent = std::make_shared<CudaEventRecordCommand>(event);
    return XEventRecordImpl(xevent, event, stream);
}

CUresult XEventRecordWithFlags(CUevent event, CUstream stream, unsigned int flags)
{
    XDEBG("XEventRecordWithFlags(event: %p, stream: %p, flags: %u)", event, stream, flags);
    CONVERT_STREAM(stream);
    if (event == nullptr) return Driver::EventRecordWithFlags(event, stream, flags);
    auto xevent = std::make_shared<CudaEventRecordWithFlagsCommand>(event, flags);
    return XEventRecordImpl(xevent, event, stream);
}

CUresult XEventRecordWithFlags_ptsz(CUevent event, CUstream stream, unsigned int flags)
{
    XDEBG("XEventRecordWithFlags_ptsz(event: %p, stream: %p, flags: %u)", event, stream, flags);
    CONVERT_STREAM_PTSZ(stream);
    if (event == nullptr) return Driver::EventRecordWithFlags(event, stream, flags);
    auto xevent = std::make_shared<CudaEventRecordWithFlagsCommand>(event, flags);
    return XEventRecordImpl(xevent, event, stream);
}

CUresult XEventQuery(CUevent event)
{
    XDEBG("XEventQuery(event: %p)", event);
    if (event == nullptr) return Driver::EventQuery(event);
    auto xevent = g_events.Get(event, nullptr);
    if (xevent == nullptr) return Driver::EventQuery(event);
    // If the event is not recorded on an XQueue, directly query from CUDA driver.
    if (xevent->GetXQueueHandle() == 0) return Driver::EventQuery(event);

    auto state = xevent->GetState();
    if (state >= kCommandStateCompleted) return CUDA_SUCCESS;
    return CUDA_ERROR_NOT_READY;
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

static CUresult XStreamWaitEventImpl(CUstream stream, CUevent event, unsigned int flags)
{
    if (event == nullptr) return Driver::StreamWaitEvent(stream, event, flags);
    CHECK_STREAM_CAPTURE(stream, Driver::StreamWaitEvent(stream, event, flags));

    auto xevent = g_events.Get(event, nullptr);
    // the event is not recorded yet
    if (xevent == nullptr) return Driver::StreamWaitEvent(stream, event, flags);

    if (stream == CU_STREAM_LEGACY) {
        // sync a event on legacy default stream
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

CUresult XStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags)
{
    XDEBG("XStreamWaitEvent(stream: %p, event: %p, flags: %u)", stream, event, flags);
    CONVERT_STREAM(stream);
    return XStreamWaitEventImpl(stream, event, flags);
}

CUresult XStreamWaitEvent_ptsz(CUstream stream, CUevent event, unsigned int flags)
{
    XDEBG("XStreamWaitEvent_ptsz(stream: %p, event: %p, flags: %u)", stream, event, flags);
    CONVERT_STREAM_PTSZ(stream);
    return XStreamWaitEventImpl(stream, event, flags);
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

static inline CUresult XStreamSynchronizeImpl(CUstream stream)
{
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return Driver::StreamSynchronize(stream);
    xq->WaitAll();
    return CUDA_SUCCESS;
}

CUresult XStreamSynchronize(CUstream stream)
{
    XDEBG("XStreamSynchronize(stream: %p)", stream);
    CHECK_STREAM(stream, Driver::StreamSynchronize(stream));
    return XStreamSynchronizeImpl(stream);
}

CUresult XStreamSynchronize_ptsz(CUstream stream)
{
    XDEBG("XStreamSynchronize_ptsz(stream: %p)", stream);
    CHECK_STREAM_PTSZ(stream, Driver::StreamSynchronize(stream));
    return XStreamSynchronizeImpl(stream);
}

static inline CUresult XStreamQueryImpl(CUstream stream)
{
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return Driver::StreamQuery(stream);

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

CUresult XStreamQuery(CUstream stream)
{
    XDEBG("XStreamQuery(stream: %p)", stream);
    CHECK_STREAM(stream, Driver::StreamQuery(stream));
    return XStreamQueryImpl(stream);
}

CUresult XStreamQuery_ptsz(CUstream stream)
{
    XDEBG("XStreamQuery_ptsz(stream: %p)", stream);
    CHECK_STREAM_PTSZ(stream, Driver::StreamQuery(stream));
    return XStreamQueryImpl(stream);
}

CUresult XCtxSynchronize()
{
    XDEBG("XCtxSynchronize()");
    XQueueManager::ForEachWaitAll();
    return Driver::CtxSynchronize();
}

static std::mutex g_single_stream_mutex;
static CUstream g_single_stream = nullptr;
static int64_t g_single_stream_ref_cnt = 0;

CUresult XStreamCreate(CUstream *stream, unsigned int flags)
{
    if (!GetCudaSingleStreamPerProcessEnabled()) {
        CUresult res = Driver::StreamCreate(stream, flags);
        if (res != CUDA_SUCCESS) return res;
        XQueueManager::AutoCreate([&](HwQueueHandle *hwq) {return CudaQueueCreate(hwq, *stream);});
        XDEBG("XStreamCreate(stream: %p, flags: 0x%x) = %d", *stream, flags, res);
        return res;
    }

    std::lock_guard<std::mutex> lock(g_single_stream_mutex);
    if (g_single_stream_ref_cnt == 0) {
        CUresult res = Driver::StreamCreate(stream, flags);
        if (res != CUDA_SUCCESS) return res;
        XQueueManager::AutoCreate([&](HwQueueHandle *hwq) {return CudaQueueCreate(hwq, *stream);});
        g_single_stream = *stream;
    }

    g_single_stream_ref_cnt++;
    *stream = g_single_stream;
    XDEBG("XStreamCreate(single stream: %p (ref: %ld), flags: 0x%x)",
          *stream, g_single_stream_ref_cnt, flags);
    return CUDA_SUCCESS;
}

CUresult XStreamCreateWithPriority(CUstream *stream, unsigned int flags, int priority)
{
    if (!GetCudaSingleStreamPerProcessEnabled()) {
        CUresult res = Driver::StreamCreateWithPriority(stream, flags, priority);
        if (res != CUDA_SUCCESS) return res;
        XQueueManager::AutoCreate([&](HwQueueHandle *hwq) {return CudaQueueCreate(hwq, *stream);});
        XDEBG("XStreamCreateWithPriority(stream: %p, flags: 0x%x, priority: %d) = %d",
              *stream, flags, priority, res);
        return res;
    }

    std::lock_guard<std::mutex> lock(g_single_stream_mutex);
    if (g_single_stream_ref_cnt == 0) {
        CUresult res = Driver::StreamCreateWithPriority(stream, flags, priority);
        if (res != CUDA_SUCCESS) return res;
        XQueueManager::AutoCreate([&](HwQueueHandle *hwq) {return CudaQueueCreate(hwq, *stream);});
        g_single_stream = *stream;
    }

    g_single_stream_ref_cnt++;
    *stream = g_single_stream;
    XDEBG("XStreamCreateWithPriority(single stream: %p (ref: %ld), flags: 0x%x, priority: %d)",
          *stream, g_single_stream_ref_cnt, flags, priority);
    return CUDA_SUCCESS;
}

CUresult XStreamDestroy(CUstream stream)
{
    if (!GetCudaSingleStreamPerProcessEnabled()) {
        XDEBG("XStreamDestroy(stream: %p)", stream);
        XQueueManager::AutoDestroy(GetHwQueueHandle(stream));
        return Driver::StreamDestroy(stream);
    }

    CUresult res = CUDA_SUCCESS;
    std::lock_guard<std::mutex> lock(g_single_stream_mutex);
    g_single_stream_ref_cnt--;
    if (g_single_stream_ref_cnt == 0) {
        XQueueManager::AutoDestroy(GetHwQueueHandle(stream));
        res = Driver::StreamDestroy(g_single_stream);
        g_single_stream = nullptr;
    }
    XDEBG("XStreamDestroy(single stream: %p (ref: %ld)) = %d",
          stream, g_single_stream_ref_cnt, res);
    return res;
}

CUresult XStreamDestroy_v2(CUstream stream)
{
    if (!GetCudaSingleStreamPerProcessEnabled()) {
        XDEBG("XStreamDestroy_v2(stream: %p)", stream);
        XQueueManager::AutoDestroy(GetHwQueueHandle(stream));
        return Driver::StreamDestroy_v2(stream);
    }

    CUresult res = CUDA_SUCCESS;
    std::lock_guard<std::mutex> lock(g_single_stream_mutex);
    g_single_stream_ref_cnt--;
    if (g_single_stream_ref_cnt == 0) {
        XQueueManager::AutoDestroy(GetHwQueueHandle(stream));
        res = Driver::StreamDestroy_v2(g_single_stream);
        g_single_stream = nullptr;
    }
    XDEBG("XStreamDestroy_v2(single stream: %p (ref: %ld)) = %d",
          stream, g_single_stream_ref_cnt, res);
    return res;
}

} // namespace xsched::cuda
