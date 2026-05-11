#include <list>
#include <dlfcn.h>
#include <chrono>
#include <thread>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <cstdlib>

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

struct shim_dim3 { unsigned int x, y, z; };

#define CALL_REAL(pfn_type, name, driver_name, ...) \
    static pfn_type p_real = nullptr; \
    if (!p_real) { \
        p_real = (pfn_type)dlsym(RTLD_NEXT, #name); \
        if (!p_real) p_real = (pfn_type)dlsym(RTLD_NEXT, #name "_v2"); \
    } \
    if (p_real) return p_real(__VA_ARGS__); \
    return Driver::driver_name(__VA_ARGS__);

// --- Shared Memory Pulse Infrastructure ---
struct VipPulse {
    std::atomic<uint64_t> last_pulse_us;
};

static VipPulse* GetVipPulseRadar() {
    static VipPulse* radar = nullptr;
    if (radar) return radar;

    int fd = shm_open("/xsched_vip_pulse", O_CREAT | O_RDWR, 0666);
    if (fd >= 0) {
        if (ftruncate(fd, sizeof(VipPulse)) == 0) {
            void* ptr = mmap(NULL, sizeof(VipPulse), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (ptr != MAP_FAILED) radar = (VipPulse*)ptr;
        }
        close(fd);
    }
    return radar;
}

static inline void PulseVipArrival() {
    auto radar = GetVipPulseRadar();
    if (radar) {
        auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        radar->last_pulse_us.store(now_us, std::memory_order_relaxed);
    }
}

static inline bool IsVipActiveOnRadar() {
    auto radar = GetVipPulseRadar();
    if (!radar) return false;
    
    auto last = radar->last_pulse_us.load(std::memory_order_relaxed);
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    
    // 如果 VIP 在 200ms 内更新过脉冲，判定 VIP 正在占用 GPU
    return (now_us - last < 200000); 
}

struct XQueueCache {
    std::shared_ptr<XQueue> xq = nullptr;
    CUstream last_stream = (CUstream)-2; 
    bool is_vip = false;
    int op_count = 0;
    int vip_burst_count = 0; 
    std::chrono::steady_clock::time_point last_op_time; 
    bool burst_active = false; 
};
static thread_local XQueueCache tl_cache;

static inline bool IsVip(std::shared_ptr<XQueue> xq) {
    if (!xq) return false;
    Priority p = xq->GetPriority();
    return (p >= 10 || p < 0);
}

static inline int GetOpThreshold() {
    static int threshold = []() {
        const char* env = getenv("XSCHED_OP_THRESHOLD");
        return env ? atoi(env) : 500;
    }();
    return threshold;
}

static inline std::shared_ptr<XQueue> GetXQueueCached(CUstream stream, bool update_vip = true) {
    if (tl_cache.last_stream == stream && tl_cache.xq) return tl_cache.xq;
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (!xq) {
        XQueueManager::AutoCreate([&](HwQueueHandle *hwq) -> XResult { return ::CudaQueueCreate(hwq, stream); });
        xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    }
    if (update_vip) {
        bool was_vip = tl_cache.is_vip;
        tl_cache.xq = xq;
        tl_cache.last_stream = stream;
        tl_cache.is_vip = IsVip(xq);
        if (was_vip != tl_cache.is_vip) {
            tl_cache.op_count = 0;
            tl_cache.vip_burst_count = 0;
            tl_cache.burst_active = false;
        }
    }
    return xq;
}

static inline void ShadowSubmit(std::shared_ptr<XQueue> xq) {
    if (!xq) return;
    auto now = std::chrono::steady_clock::now();
    
    if (tl_cache.is_vip) {
        // --- VIP 逻辑：高频喊话，维持威慑力 ---
        auto idle_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - tl_cache.last_op_time).count();
        if (idle_ms > 50) tl_cache.vip_burst_count = 0;
        tl_cache.last_op_time = now;

        // 更新雷达脉冲：每一笔发射都喊话，确保 Normal 进程瞬间感应
        PulseVipArrival();

        // XServer 报账逻辑维持，建立初始控制权
        if (tl_cache.vip_burst_count < 10) {
            xq->Submit(std::make_shared<CudaRuntimeLaunchCommand>());
            tl_cache.vip_burst_count++;
            tl_cache.op_count = 0;
            return;
        }
        
        // VIP 心跳极致稀疏化，保护 L40S 原生性能
        if (++tl_cache.op_count >= 3000) {
            xq->Submit(std::make_shared<CudaRuntimeLaunchCommand>());
            tl_cache.op_count = 0;
        }
        return;
    }

    // --- Normal 进程：基于雷达的高速自适应 ---
    if (IsVipActiveOnRadar()) {
        // 发现 VIP 脉冲：瞬间进入战时阻塞模式，保障抢占
        xq->Submit(std::make_shared<CudaRuntimeLaunchCommand>());
        xq->WaitAll(); 
        tl_cache.op_count = 0;
        tl_cache.burst_active = true;
    } else {
        // VIP 不在场：完全静默执行。每隔一定阈值才报一笔维持状态。
        // 这能让 37ms 的 Embedding 回归原生性能，因为其推理全程可能 0-IPC。
        if (++tl_cache.op_count >= GetOpThreshold()) {
            xq->Submit(std::make_shared<CudaRuntimeLaunchCommand>());
            tl_cache.op_count = 0;
            tl_cache.burst_active = true;
        }
    }
}

void WaitBlockingXQueues() { XQueueManager::ForEachWaitAll(); }

CUresult XLaunchKernel(CUfunction f, unsigned int gdx, unsigned int gdy, unsigned int gdz,
                       unsigned int bdx, unsigned int bdy, unsigned int bdz,
                       unsigned int shmem, CUstream stream, void **params, void **extra)
{
    ShadowSubmit(GetXQueueCached(stream));
    typedef CUresult (*pfn)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void **, void **);
    CALL_REAL(pfn, cuLaunchKernel, LaunchKernel, f, gdx, gdy, gdz, bdx, bdy, bdz, shmem, stream, params, extra);
}

CUresult XLaunchKernelRuntime(const void *func, unsigned int gdx, unsigned int gdy, unsigned int gdz,
                              unsigned int bdx, unsigned int bdy, unsigned int bdz,
                              size_t shmem, void **args, CUstream stream)
{
    ShadowSubmit(GetXQueueCached(stream));
    typedef int (*pfn)(const void *, shim_dim3, shim_dim3, void **, size_t, CUstream);
    static pfn p_real = nullptr;
    if (!p_real) {
        p_real = (pfn)dlvsym(RTLD_NEXT, "cudaLaunchKernel", "GLIBC_2.2.5");
        if (!p_real) p_real = (pfn)dlsym(RTLD_NEXT, "cudaLaunchKernel");
    }
    if (p_real) {
        shim_dim3 gd = {gdx, gdy, gdz}, bd = {bdx, bdy, bdz};
        return (CUresult)p_real(func, gd, bd, args, shmem, stream);
    }
    return CUDA_ERROR_NOT_FOUND;
}

CUresult XStreamSynchronize(CUstream stream) {
    auto xq = GetXQueueCached(stream, false);
    // 只有在检测到 VIP 活跃或刚刚报过账的情况下才执行 WaitAll
    if (xq && !tl_cache.is_vip && (IsVipActiveOnRadar() || tl_cache.burst_active)) {
        xq->WaitAll(); 
    }
    tl_cache.burst_active = false; 
    typedef CUresult (*pfn)(CUstream);
    CALL_REAL(pfn, cuStreamSynchronize, StreamSynchronize, stream);
}

CUresult XMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    typedef CUresult (*pfn)(void *, CUdeviceptr, size_t);
    CALL_REAL(pfn, cuMemcpyDtoH, MemcpyDtoH_v2, dstHost, srcDevice, ByteCount);
}

CUresult XMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    typedef CUresult (*pfn)(CUdeviceptr, const void *, size_t);
    CALL_REAL(pfn, cuMemcpyHtoD, MemcpyHtoD_v2, dstDevice, srcHost, ByteCount);
}

CUresult XMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    typedef CUresult (*pfn)(CUdeviceptr, CUdeviceptr, size_t);
    CALL_REAL(pfn, cuMemcpyDtoD, MemcpyDtoD_v2, dstDevice, srcDevice, ByteCount);
}

CUresult XMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
    auto xq = GetXQueueCached(hStream, false);
    if (xq && !tl_cache.is_vip && IsVipActiveOnRadar()) { xq->Submit(std::make_shared<CudaRuntimeLaunchCommand>()); xq->WaitAll(); }
    typedef CUresult (*pfn)(CUdeviceptr, const void *, size_t, CUstream);
    CALL_REAL(pfn, cuMemcpyHtoDAsync, MemcpyHtoDAsync_v2, dstDevice, srcHost, ByteCount, hStream);
}

CUresult XMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    auto xq = GetXQueueCached(hStream, false);
    if (xq && !tl_cache.is_vip && IsVipActiveOnRadar()) { xq->Submit(std::make_shared<CudaRuntimeLaunchCommand>()); xq->WaitAll(); }
    typedef CUresult (*pfn)(void *, CUdeviceptr, size_t, CUstream);
    CALL_REAL(pfn, cuMemcpyDtoHAsync, MemcpyDtoHAsync_v2, dstHost, srcDevice, ByteCount, hStream);
}

CUresult XEventSynchronize(CUevent event) {
    auto xev = g_events.Get(event, nullptr);
    if (xev) xev->Wait();
    typedef CUresult (*pfn)(CUevent);
    CALL_REAL(pfn, cuEventSynchronize, EventSynchronize, event);
}

CUresult XEventQuery(CUevent event) {
    typedef CUresult (*pfn)(CUevent);
    static pfn p_real = (pfn)dlsym(RTLD_NEXT, "cuEventQuery");
    return p_real ? p_real(event) : Driver::EventQuery(event);
}

CUresult XStreamQuery(CUstream stream) {
    typedef CUresult (*pfn)(CUstream);
    static pfn p_real = (pfn)dlsym(RTLD_NEXT, "cuStreamQuery");
    return p_real ? p_real(stream) : Driver::StreamQuery(stream);
}

CUresult XEventRecord(CUevent event, CUstream stream) {
    auto xq = GetXQueueCached(stream, false);
    auto xev = std::make_shared<CudaEventRecordCommand>(event);
    if (xq && !tl_cache.is_vip && IsVipActiveOnRadar()) { xq->Submit(xev); xq->WaitAll(); }
    g_events.Add(event, xev);
    typedef CUresult (*pfn)(CUevent, CUstream);
    CALL_REAL(pfn, cuEventRecord, EventRecord, event, stream);
}

CUresult XEventRecordWithFlags(CUevent event, CUstream stream, unsigned int flags) {
    auto xq = GetXQueueCached(stream, false);
    auto xev = std::make_shared<CudaEventRecordWithFlagsCommand>(event, flags);
    if (xq && !tl_cache.is_vip && IsVipActiveOnRadar()) { xq->Submit(xev); xq->WaitAll(); }
    g_events.Add(event, xev);
    typedef CUresult (*pfn)(CUevent, CUstream, unsigned int);
    CALL_REAL(pfn, cuEventRecordWithFlags, EventRecordWithFlags, event, stream, flags);
}

CUresult XStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags) {
    auto xq = GetXQueueCached(stream, false);
    auto xev = g_events.Get(event, nullptr);
    if (xev && xq && !tl_cache.is_vip && IsVipActiveOnRadar()) { xq->Submit(std::make_shared<CudaEventWaitCommand>(xev, flags)); xq->WaitAll(); }
    typedef CUresult (*pfn)(CUstream, CUevent, unsigned int);
    CALL_REAL(pfn, cuStreamWaitEvent, StreamWaitEvent, stream, event, flags);
}

CUresult XCtxSynchronize() {
    XQueueManager::ForEachWaitAll(); 
    typedef CUresult (*pfn)();
    CALL_REAL(pfn, cuCtxSynchronize, CtxSynchronize);
}

CUresult XMemcpyDtoH(void *dstHost, CUdeviceptr_v1 srcDevice, unsigned int ByteCount) {
    return XMemcpyDtoH_v2(dstHost, (CUdeviceptr)srcDevice, (size_t)ByteCount);
}
CUresult XMemcpyHtoD(CUdeviceptr_v1 dstDevice, const void *srcHost, unsigned int ByteCount) {
    return XMemcpyHtoD_v2((CUdeviceptr)dstDevice, srcHost, (size_t)ByteCount);
}

CUresult XEventDestroy(CUevent event) {
    if (event) g_events.DoThenDel(event, nullptr, [](auto xev) { xev->DestroyEvent(); });
    return Driver::EventDestroy(event);
}
CUresult XEventDestroy_v2(CUevent event) {
    if (event) g_events.DoThenDel(event, nullptr, [](auto xev) { xev->DestroyEvent(); });
    return Driver::EventDestroy_v2(event);
}
CUresult XStreamCreate(CUstream *stream, unsigned int flags) {
    CUresult res = Driver::StreamCreate(stream, flags);
    if (res == CUDA_SUCCESS) XQueueManager::AutoCreate([&](HwQueueHandle *hwq) -> XResult { return ::CudaQueueCreate(hwq, *stream); });
    return res;
}
CUresult XStreamCreateWithPriority(CUstream *stream, unsigned int flags, int priority) {
    CUresult res = Driver::StreamCreateWithPriority(stream, flags, priority);
    if (res == CUDA_SUCCESS) XQueueManager::AutoCreate([&](HwQueueHandle *hwq) -> XResult { return ::CudaQueueCreate(hwq, *stream); });
    return res;
}
CUresult XStreamDestroy(CUstream stream) {
    XQueueManager::AutoDestroy(GetHwQueueHandle(stream)); return Driver::StreamDestroy(stream);
}
CUresult XStreamDestroy_v2(CUstream stream) {
    XQueueManager::AutoDestroy(GetHwQueueHandle(stream)); return Driver::StreamDestroy_v2(stream);
}
CUresult XLaunchHostFunc(CUstream stream, CUhostFn fn, void *data) {
    auto xq = GetXQueueCached(stream, false);
    if (xq) { xq->Submit(std::make_shared<CudaHostFuncCommand>(fn, data)); }
    typedef CUresult (*pfn)(CUstream, CUhostFn, void*);
    CALL_REAL(pfn, cuLaunchHostFunc, LaunchHostFunc, stream, fn, data);
}

} // namespace xsched::cuda
