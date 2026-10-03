#include <list>
#include <mutex>
#include <atomic>
#include <functional>
#include <unordered_map>

#include "xsched/utils/log.h"
#include "xsched/utils/map.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/cuda/hal.h"
#include "xsched/cuda/shim/shim.h"
#include "xsched/cuda/shim/context_registry.h"
#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/common/levels.h"
#include "xsched/cuda/hal/common/options.h"
#include "xsched/cuda/hal/level1/cuda_queue.h"
#include "xsched/cuda/hal/common/cuda_command.h"
#include "xsched/cuda/hal/common/event_pool.h"

using namespace xsched::preempt;

namespace xsched::cuda
{

static utils::ObjectMap<CUevent, std::shared_ptr<CudaEventRecordCommand>> g_events;

namespace
{
/// @brief Per-thread storage of the per-thread default streams (PTDS).
/// CUDA does not provide any API to enumerate or destroy the per-thread
/// default streams, so XSched tracks and destroys them on thread exit.
struct PtdsEntry
{
    CUstream stream = nullptr;
    uint64_t destroy_gen = 0;
};

/// @brief An orphaned PTDS stream awaiting deferred teardown.
struct PtdsOrphan
{
    CUcontext ctx = nullptr;
    CUstream stream = nullptr;
};

/// Orphan queue for deferred PTDS teardown.
///
/// CRITICAL (Windows): a thread_local destructor runs in the thread-exit
/// path (for MinGW via DLL_THREAD_DETACH) while the loader lock is held.
/// Blocking operations performed there — XQueue wait/destroy and especially
/// joining the launch worker thread — can deadlock the whole process
/// against LdrShutdownThread of the worker. Therefore the destructor below
/// only performs lock-free bookkeeping (unregister + enqueue), and the
/// actual teardown is drained later from a normal thread context.
std::mutex g_ptds_orphan_mutex;
std::vector<PtdsOrphan> g_ptds_orphans;
std::atomic<int64_t> g_ptds_orphan_cnt {0};

void EnqueuePtdsOrphan(CUcontext ctx, CUstream stream)
{
    std::lock_guard<std::mutex> lock(g_ptds_orphan_mutex);
    g_ptds_orphans.push_back({ctx, stream});
    g_ptds_orphan_cnt.fetch_add(1, std::memory_order_relaxed);
}

/// @brief Tear down orphaned PTDS streams. MUST only be called from a normal
/// thread context (never from a thread-exit / DLL_THREAD_DETACH callback).
/// @param ctx If not nullptr, only orphans of this context are drained.
void DrainPtdsOrphans(CUcontext ctx)
{
    if (g_ptds_orphan_cnt.load(std::memory_order_relaxed) == 0) return;

    std::vector<PtdsOrphan> orphans;
    {
        std::lock_guard<std::mutex> lock(g_ptds_orphan_mutex);
        if (g_ptds_orphans.empty()) return;
        if (ctx == nullptr) {
            orphans.swap(g_ptds_orphans);
            g_ptds_orphan_cnt.store(0, std::memory_order_relaxed);
        } else {
            auto it = g_ptds_orphans.begin();
            while (it != g_ptds_orphans.end()) {
                if (it->ctx == ctx) {
                    orphans.push_back(*it);
                    it = g_ptds_orphans.erase(it);
                    g_ptds_orphan_cnt.fetch_sub(1, std::memory_order_relaxed);
                } else {
                    ++it;
                }
            }
        }
    }

    CUcontext prev_ctx = nullptr;
    Driver::CtxGetCurrent(&prev_ctx);
    for (auto &orphan : orphans)
    {
        CUstream stream = orphan.stream;
        if (stream == nullptr) continue;

        /// Guard against handle recycling: an orphan that was enqueued after
        /// its context had already been destroyed may collide with a NEW
        /// stream that the driver allocated at the same address. Only proceed
        /// when the stream is still valid and still belongs to its context.
        CUcontext stream_ctx = nullptr;
        if (Driver::StreamGetCtx(stream, &stream_ctx) != CUDA_SUCCESS) {
            XWARN("stale PTDS orphan %p (ctx: %p) skipped: stream is gone", stream, orphan.ctx);
            continue;
        }
        if (stream_ctx != orphan.ctx) {
            XWARN("stale PTDS orphan %p skipped: handle reused by ctx %p", stream, stream_ctx);
            continue;
        }

        HwQueueHandle hwq_h = GetHwQueueHandle(stream);
        auto xq = HwQueueManager::GetXQueue(hwq_h);
        if (xq != nullptr) xq->WaitAll();
        XQueueManager::AutoDestroy(hwq_h);
        CudaContextRegistry::Unregister(orphan.ctx, hwq_h);
        if (Driver::CtxSetCurrent(orphan.ctx) == CUDA_SUCCESS) {
            Driver::StreamDestroy(stream);
        }
        XDEBG("PTDS stream %p cleaned up (deferred, ctx: %p)", stream, orphan.ctx);
    }
    // Restore the caller's current context (draining is a side effect).
    Driver::CtxSetCurrent(prev_ctx);
}

struct PtdsMap
{
    std::unordered_map<CUcontext, PtdsEntry> map;

    ~PtdsMap()
    {
        if (map.empty()) return;
        /// CRITICAL (Windows): this destructor runs in the thread-exit path
        /// (DLL_THREAD_DETACH, loader lock held). Only non-blocking,
        /// pure in-memory operations are allowed here. The heavyweight
        /// teardown (XQueue wait/destroy, launch worker join) is deferred
        /// to DrainPtdsOrphans() in a safe context.
        CUcontext probe = nullptr;
        if (Driver::CtxGetCurrent(&probe) != CUDA_SUCCESS) return;

        for (auto &kv : map)
        {
            CUstream stream = kv.second.stream;
            if (stream == nullptr) continue;
            CudaContextRegistry::Unregister(kv.first, GetHwQueueHandle(stream));
            EnqueuePtdsOrphan(kv.first, stream);
        }
        map.clear();
    }
};
} // anonymous namespace

CUstream GetPTDS()
{
    /// Each CUDA context should have its own per-thread default stream.
    CUcontext ctx = nullptr;
    if (Driver::CtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == nullptr) {
        return nullptr;
    }

    // Safe context: opportunistically tear down PTDS streams of exited threads.
    DrainPtdsOrphans(nullptr);

    static thread_local PtdsMap ptds;

    auto it = ptds.map.find(ctx);
    if (it != ptds.map.end()) {
        if (it->second.destroy_gen == CudaContextRegistry::DestroyGeneration()) {
            return it->second.stream;
        }
        // At least one context has been destroyed since this entry was
        // created. The CUcontext pointer may have been recycled by the
        // driver for a new context: verify that the cached stream is still
        // valid and actually belongs to ctx.
        CUcontext stream_ctx = nullptr;
        if (Driver::StreamGetCtx(it->second.stream, &stream_ctx) == CUDA_SUCCESS &&
            stream_ctx == ctx) {
            it->second.destroy_gen = CudaContextRegistry::DestroyGeneration();
            return it->second.stream;
        }
        ptds.map.erase(it);
    }

    CUstream stream = nullptr;
    if (Driver::StreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
        return nullptr;
    }

    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) {return CudaQueueCreate(hwq, stream);});
    CudaContextRegistry::Register(ctx, GetHwQueueHandle(stream));
    ptds.map[ctx] = {stream, CudaContextRegistry::DestroyGeneration()};
    return stream;
}

void WaitBlockingXQueues()
{
    CUcontext ctx = nullptr;
    if (Driver::CtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == nullptr) return;
    auto handles = CudaContextRegistry::GetSnapshot(ctx);

    std::list<std::shared_ptr<XQueueWaitAllCommand>> wait_cmds;
    for (auto hwq_h : handles) {
        auto xq = HwQueueManager::GetXQueue(hwq_h);
        if (xq == nullptr) continue;
        auto hwq = xq->GetHwQueue();
        auto cuda_q = std::dynamic_pointer_cast<CudaQueueLv1>(hwq);
        if (cuda_q == nullptr) continue;
        if (cuda_q->GetStreamFlags() & CU_STREAM_NON_BLOCKING) continue;
        auto wait_cmd = xq->SubmitWaitAll();
        if (wait_cmd == nullptr) continue;
        wait_cmds.push_back(wait_cmd);
    }
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

CUresult XLaunchCooperativeKernel(CUfunction f,
                                  unsigned int gdx, unsigned int gdy, unsigned int gdz,
                                  unsigned int bdx, unsigned int bdy, unsigned int bdz,
                                  unsigned int shmem, CUstream stream, void **params)
{
    XDEBG("XLaunchCooperativeKernel(func: %p, stream: %p, grid: [%u, %u, %u], "
          "block: [%u, %u, %u], shm: %u, params: %p)",
          f, stream, gdx, gdy, gdz, bdx, bdy, bdz, shmem, params);
    CHECK_STREAM(stream, DirectLaunch(std::make_shared<CudaLaunchCooperativeKernelCommand>(
        f, gdx, gdy, gdz, bdx, bdy, bdz, shmem, params, false), stream));
    return XLaunchKernelImpl<CudaLaunchCooperativeKernelCommand>(
        stream, f, gdx, gdy, gdz, bdx, bdy, bdz, shmem, params);
}

CUresult XLaunchCooperativeKernel_ptsz(CUfunction f,
                                       unsigned int gdx, unsigned int gdy, unsigned int gdz,
                                       unsigned int bdx, unsigned int bdy, unsigned int bdz,
                                       unsigned int shmem, CUstream stream, void **params)
{
    XDEBG("XLaunchCooperativeKernel_ptsz(func: %p, stream: %p, grid: [%u, %u, %u], "
          "block: [%u, %u, %u], shm: %u, params: %p)",
          f, stream, gdx, gdy, gdz, bdx, bdy, bdz, shmem, params);
    CHECK_STREAM_PTSZ(stream, DirectLaunch(std::make_shared<CudaLaunchCooperativeKernelCommand>(
        f, gdx, gdy, gdz, bdx, bdy, bdz, shmem, params, false), stream));
    return XLaunchKernelImpl<CudaLaunchCooperativeKernelCommand>(
        stream, f, gdx, gdy, gdz, bdx, bdy, bdz, shmem, params);
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
    /// Optimized: only wait for commands in the current context.
    /// CUDA semantics: cuMemFree waits for commands in the current context.
    CUcontext ctx = nullptr;
    if (Driver::CtxGetCurrent(&ctx) == CUDA_SUCCESS && ctx != nullptr) {
        CudaContextRegistry::WaitAllInContext(ctx);
    }
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
    CUcontext ctx = nullptr;
    if (Driver::CtxGetCurrent(&ctx) == CUDA_SUCCESS && ctx != nullptr) {
        CudaContextRegistry::WaitAllInContext(ctx);
    }
    return Driver::CtxSynchronize();
}

struct SingleStreamInfo {
    CUstream stream = nullptr;
    int64_t ref_cnt = 0;
};
static std::mutex g_single_stream_mutex;
static std::unordered_map<CUcontext, SingleStreamInfo> g_single_streams;

CUresult XStreamCreate(CUstream *stream, unsigned int flags)
{
    if (!GetCudaSingleStreamPerProcessEnabled()) {
        CUresult res = Driver::StreamCreate(stream, flags);
        if (res != CUDA_SUCCESS) return res;
        XQueueManager::AutoCreate([&](HwQueueHandle *hwq) {return CudaQueueCreate(hwq, *stream);});
        CUcontext ctx = nullptr;
        if (Driver::StreamGetCtx(*stream, &ctx) == CUDA_SUCCESS && ctx != nullptr) {
            CudaContextRegistry::Register(ctx, GetHwQueueHandle(*stream));
        }
        XDEBG("XStreamCreate(stream: %p, flags: 0x%x) = %d", *stream, flags, res);
        return res;
    }

    CUcontext ctx = nullptr;
    Driver::CtxGetCurrent(&ctx);
    if (ctx == nullptr) {
        // No current context: hand the call over to the driver, which
        // returns the proper error (CUDA_ERROR_INVALID_CONTEXT).
        return Driver::StreamCreate(stream, flags);
    }
    std::lock_guard<std::mutex> lock(g_single_stream_mutex);
    auto &info = g_single_streams[ctx];
    if (info.ref_cnt == 0) {
        CUresult res = Driver::StreamCreate(stream, flags);
        if (res != CUDA_SUCCESS) return res;
        XQueueManager::AutoCreate([&](HwQueueHandle *hwq) {return CudaQueueCreate(hwq, *stream);});
        CudaContextRegistry::Register(ctx, GetHwQueueHandle(*stream));
        info.stream = *stream;
    } else if (flags != 0) {
        XWARN("XStreamCreate: flags 0x%x ignored in single-stream mode (stream %p already created)",
              flags, info.stream);
    }

    info.ref_cnt++;
    *stream = info.stream;
    XDEBG("XStreamCreate(single stream: %p (ref: %lld), ctx: %p, flags: 0x%x)",
          *stream, (long long)info.ref_cnt, ctx, flags);
    return CUDA_SUCCESS;
}

CUresult XStreamCreateWithPriority(CUstream *stream, unsigned int flags, int priority)
{
    if (!GetCudaSingleStreamPerProcessEnabled()) {
        CUresult res = Driver::StreamCreateWithPriority(stream, flags, priority);
        if (res != CUDA_SUCCESS) return res;
        XQueueManager::AutoCreate([&](HwQueueHandle *hwq) {return CudaQueueCreate(hwq, *stream);});
        CUcontext ctx = nullptr;
        if (Driver::StreamGetCtx(*stream, &ctx) == CUDA_SUCCESS && ctx != nullptr) {
            CudaContextRegistry::Register(ctx, GetHwQueueHandle(*stream));
        }
        XDEBG("XStreamCreateWithPriority(stream: %p, flags: 0x%x, priority: %d) = %d",
              *stream, flags, priority, res);
        return res;
    }

    CUcontext ctx = nullptr;
    Driver::CtxGetCurrent(&ctx);
    if (ctx == nullptr) {
        // No current context: hand the call over to the driver, which
        // returns the proper error (CUDA_ERROR_INVALID_CONTEXT).
        return Driver::StreamCreateWithPriority(stream, flags, priority);
    }
    std::lock_guard<std::mutex> lock(g_single_stream_mutex);
    auto &info = g_single_streams[ctx];
    if (info.ref_cnt == 0) {
        CUresult res = Driver::StreamCreateWithPriority(stream, flags, priority);
        if (res != CUDA_SUCCESS) return res;
        XQueueManager::AutoCreate([&](HwQueueHandle *hwq) {return CudaQueueCreate(hwq, *stream);});
        CudaContextRegistry::Register(ctx, GetHwQueueHandle(*stream));
        info.stream = *stream;
    } else if (flags != 0 || priority != 0) {
        XWARN("XStreamCreateWithPriority: flags 0x%x, priority %d ignored in single-stream mode",
              flags, priority);
    }

    info.ref_cnt++;
    *stream = info.stream;
    XDEBG("XStreamCreateWithPriority(single stream: %p (ref: %lld), ctx: %p, flags: 0x%x, priority: %d)",
          *stream, (long long)info.ref_cnt, ctx, flags, priority);
    return CUDA_SUCCESS;
}

CUresult XStreamDestroy(CUstream stream)
{
    if (!GetCudaSingleStreamPerProcessEnabled()) {
        XDEBG("XStreamDestroy(stream: %p)", stream);
        CUcontext ctx = nullptr;
        Driver::StreamGetCtx(stream, &ctx);
        CudaContextRegistry::Unregister(ctx, GetHwQueueHandle(stream));
        XQueueManager::AutoDestroy(GetHwQueueHandle(stream));
        return Driver::StreamDestroy(stream);
    }

    CUresult res = CUDA_SUCCESS;
    std::lock_guard<std::mutex> lock(g_single_stream_mutex);
    CUcontext ctx = nullptr;
    Driver::StreamGetCtx(stream, &ctx);
    auto it = g_single_streams.find(ctx);
    if (it != g_single_streams.end()) {
        it->second.ref_cnt--;
        if (it->second.ref_cnt <= 0) {
            CudaContextRegistry::Unregister(ctx, GetHwQueueHandle(stream));
            XQueueManager::AutoDestroy(GetHwQueueHandle(stream));
            res = Driver::StreamDestroy(it->second.stream);
            g_single_streams.erase(it);
        }
    }
    XDEBG("XStreamDestroy(stream: %p, ctx: %p) = %d", stream, ctx, res);
    return res;
}

CUresult XStreamDestroy_v2(CUstream stream)
{
    if (!GetCudaSingleStreamPerProcessEnabled()) {
        XDEBG("XStreamDestroy_v2(stream: %p)", stream);
        CUcontext ctx = nullptr;
        Driver::StreamGetCtx(stream, &ctx);
        CudaContextRegistry::Unregister(ctx, GetHwQueueHandle(stream));
        XQueueManager::AutoDestroy(GetHwQueueHandle(stream));
        return Driver::StreamDestroy_v2(stream);
    }

    CUresult res = CUDA_SUCCESS;
    std::lock_guard<std::mutex> lock(g_single_stream_mutex);
    CUcontext ctx = nullptr;
    Driver::StreamGetCtx(stream, &ctx);
    auto it = g_single_streams.find(ctx);
    if (it != g_single_streams.end()) {
        it->second.ref_cnt--;
        if (it->second.ref_cnt <= 0) {
            CudaContextRegistry::Unregister(ctx, GetHwQueueHandle(stream));
            XQueueManager::AutoDestroy(GetHwQueueHandle(stream));
            res = Driver::StreamDestroy_v2(it->second.stream);
            g_single_streams.erase(it);
        }
    }
    XDEBG("XStreamDestroy_v2(stream: %p, ctx: %p) = %d", stream, ctx, res);
    return res;
}

/// @brief Drain and clean up all XSched resources associated with a context
/// right before its physical destruction.
static void DrainContextBeforeDestroy(CUcontext ctx)
{
    if (ctx == nullptr) return;
    CudaContextRegistry::DrainAndClearContext(ctx); // wait, destroy XQueues, clear registry
    CudaEventPool::Clear(ctx);                      // destroy cached CUDA events
    DrainPtdsOrphans(ctx);                          // tear down orphaned PTDS streams
    std::lock_guard<std::mutex> lock(g_single_stream_mutex);
    g_single_streams.erase(ctx);                    // drop single-stream mode state
}

CUresult XCtxDestroy(CUcontext ctx)
{
    XDEBG("XCtxDestroy(ctx: %p)", ctx);
    DrainContextBeforeDestroy(ctx);
    return Driver::CtxDestroy(ctx);
}

CUresult XCtxDestroy_v2(CUcontext ctx)
{
    XDEBG("XCtxDestroy_v2(ctx: %p)", ctx);
    DrainContextBeforeDestroy(ctx);
    return Driver::CtxDestroy_v2(ctx);
}

CUresult XDevicePrimaryCtxReset(CUdevice dev)
{
    XDEBG("XDevicePrimaryCtxReset(dev: %d)", dev);
    CUcontext ctx = nullptr;
    if (Driver::DevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS && ctx != nullptr) {
        DrainContextBeforeDestroy(ctx);
        Driver::DevicePrimaryCtxRelease_v2(dev); // balance the retain above
    }
    return Driver::DevicePrimaryCtxReset(dev);
}

CUresult XDevicePrimaryCtxReset_v2(CUdevice dev)
{
    XDEBG("XDevicePrimaryCtxReset_v2(dev: %d)", dev);
    CUcontext ctx = nullptr;
    if (Driver::DevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS && ctx != nullptr) {
        DrainContextBeforeDestroy(ctx);
        Driver::DevicePrimaryCtxRelease_v2(dev); // balance the retain above
    }
    return Driver::DevicePrimaryCtxReset_v2(dev);
}

CUresult XDevicePrimaryCtxRelease(CUdevice dev)
{
    XDEBG("XDevicePrimaryCtxRelease(dev: %d)", dev);
    /// When the reference count drops to zero, the driver deinitializes the
    /// primary context — a destruction path XSched cannot observe precisely
    /// (the release may or may not be the last one). Advancing the destroy
    /// generation makes every cached per-thread default stream revalidate
    /// itself lazily on next use (see GetPTDS), preventing stale-stream reuse
    /// even if the driver later recycles the CUcontext pointer.
    CudaContextRegistry::BumpDestroyGeneration();
    return Driver::DevicePrimaryCtxRelease(dev);
}

CUresult XDevicePrimaryCtxRelease_v2(CUdevice dev)
{
    XDEBG("XDevicePrimaryCtxRelease_v2(dev: %d)", dev);
    CudaContextRegistry::BumpDestroyGeneration();
    return Driver::DevicePrimaryCtxRelease_v2(dev);
}

CUresult XEventElapsedTime(float *ms, CUevent start, CUevent end)
{
    XDEBG("XEventElapsedTime(ms: %p, start: %p, end: %p)", ms, start, end);
    if (start != nullptr) {
        auto xstart = g_events.Get(start, nullptr);
        if (xstart != nullptr) xstart->Wait();
    }
    if (end != nullptr) {
        auto xend = g_events.Get(end, nullptr);
        if (xend != nullptr) xend->Wait();
    }
    return Driver::EventElapsedTime(ms, start, end);
}

CUresult XEventElapsedTime_v2(float *ms, CUevent start, CUevent end)
{
    XDEBG("XEventElapsedTime_v2(ms: %p, start: %p, end: %p)", ms, start, end);
    if (start != nullptr) {
        auto xstart = g_events.Get(start, nullptr);
        if (xstart != nullptr) xstart->Wait();
    }
    if (end != nullptr) {
        auto xend = g_events.Get(end, nullptr);
        if (xend != nullptr) xend->Wait();
    }
    return Driver::EventElapsedTime_v2(ms, start, end);
}

/// @note Synchronous memcpy with per-thread default stream semantics:
/// the copy is issued on the calling thread's per-thread default stream.
/// Wait for the commands already submitted on that stream's XQueue before
/// handing the copy over to the driver; implicit synchronization with the
/// legacy stream (if any) is preserved by the driver's _ptds variant.
static void WaitPTDSXQueue()
{
    CUstream ptds = GetPTDS();
    if (ptds == nullptr) return;
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(ptds));
    if (xq != nullptr) xq->WaitAll();
}

CUresult XMemcpyHtoD_v2_ptds(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)
{
    XDEBG("XMemcpyHtoD_v2_ptds(dst: %p, src: %p, size: %zu)", (void *)dstDevice, srcHost, ByteCount);
    WaitPTDSXQueue();
    return Driver::MemcpyHtoD_v2_ptds(dstDevice, srcHost, ByteCount);
}

CUresult XMemcpyDtoH_v2_ptds(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    XDEBG("XMemcpyDtoH_v2_ptds(dst: %p, src: %p, size: %zu)", dstHost, (void *)srcDevice, ByteCount);
    WaitPTDSXQueue();
    return Driver::MemcpyDtoH_v2_ptds(dstHost, srcDevice, ByteCount);
}

CUresult XMemcpyDtoD_v2_ptds(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
{
    XDEBG("XMemcpyDtoD_v2_ptds(dst: %p, src: %p, size: %zu)", (void *)dstDevice, (void *)srcDevice, ByteCount);
    WaitPTDSXQueue();
    return Driver::MemcpyDtoD_v2_ptds(dstDevice, srcDevice, ByteCount);
}

CUresult XMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)
{
    XDEBG("XMemcpyHtoD_v2(dst: %p, src: %p, size: %zu)", (void *)dstDevice, srcHost, ByteCount);
    WaitBlockingXQueues();
    return Driver::MemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
}

CUresult XMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    XDEBG("XMemcpyDtoH_v2(dst: %p, src: %p, size: %zu)", dstHost, (void *)srcDevice, ByteCount);
    WaitBlockingXQueues();
    return Driver::MemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
}

CUresult XMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
{
    XDEBG("XMemcpyDtoD_v2(dst: %p, src: %p, size: %zu)", (void *)dstDevice, (void *)srcDevice, ByteCount);
    WaitBlockingXQueues();
    return Driver::MemcpyDtoD_v2(dstDevice, srcDevice, ByteCount);
}

} // namespace xsched::cuda
