#include <list>
#include <memory>
#include <atomic>
#include <algorithm>

#include "xsched/utils/log.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/cuda/shim/context_registry.h"

namespace xsched::cuda
{

std::shared_mutex CudaContextRegistry::mtx_;
std::unordered_map<CUcontext, std::vector<HwQueueHandle>> CudaContextRegistry::registry_;

/// Incremented on every context destroy; see DestroyGeneration().
static std::atomic<uint64_t> g_destroy_gen {0};

uint64_t CudaContextRegistry::DestroyGeneration()
{
    return g_destroy_gen.load(std::memory_order_acquire);
}

void CudaContextRegistry::BumpDestroyGeneration()
{
    g_destroy_gen.fetch_add(1, std::memory_order_acq_rel);
}

void CudaContextRegistry::Register(CUcontext ctx, HwQueueHandle hwq_h)
{
    if (ctx == nullptr || hwq_h == 0) return;
    std::unique_lock lock(mtx_);
    auto &vec = registry_[ctx];
    if (std::find(vec.begin(), vec.end(), hwq_h) == vec.end()) {
        vec.push_back(hwq_h);
        XDEBG("CudaContextRegistry: registered hwq 0x%lx to context %p (count: %zu)",
              (unsigned long)hwq_h, ctx, vec.size());
    }
}

void CudaContextRegistry::Unregister(CUcontext ctx, HwQueueHandle hwq_h)
{
    if (hwq_h == 0) return;
    std::unique_lock lock(mtx_);

    bool found = false;
    if (ctx != nullptr) {
        auto it = registry_.find(ctx);
        if (it != registry_.end()) {
            auto &vec = it->second;
            auto pos = std::find(vec.begin(), vec.end(), hwq_h);
            if (pos != vec.end()) {
                vec.erase(pos);
                if (vec.empty()) registry_.erase(it);
                found = true;
                XDEBG("CudaContextRegistry: unregistered hwq 0x%lx from context %p",
                      (unsigned long)hwq_h, ctx);
            }
        }
    }

    // Defensive fallback: if not found under given ctx, scan all contexts
    if (!found) {
        for (auto it = registry_.begin(); it != registry_.end(); ) {
            auto &vec = it->second;
            auto pos = std::find(vec.begin(), vec.end(), hwq_h);
            if (pos != vec.end()) {
                vec.erase(pos);
                XDEBG("CudaContextRegistry: unregistered hwq 0x%lx from context %p (scan fallback)",
                      (unsigned long)hwq_h, it->first);
            }
            if (vec.empty()) {
                it = registry_.erase(it);
            } else {
                ++it;
            }
        }
    }
}

std::vector<HwQueueHandle> CudaContextRegistry::GetSnapshot(CUcontext ctx)
{
    if (ctx == nullptr) return {};
    std::shared_lock lock(mtx_);
    auto it = registry_.find(ctx);
    if (it == registry_.end()) return {};
    return it->second;
}

void CudaContextRegistry::WaitAllInContext(CUcontext ctx)
{
    if (ctx == nullptr) return;
    auto snapshot = GetSnapshot(ctx);
    if (snapshot.empty()) return;

    std::list<std::shared_ptr<preempt::XQueueWaitAllCommand>> wait_cmds;
    for (auto hwq_h : snapshot) {
        auto xq = preempt::HwQueueManager::GetXQueue(hwq_h);
        if (xq == nullptr) continue;
        auto wait_cmd = xq->SubmitWaitAll();
        if (wait_cmd != nullptr) {
            wait_cmds.push_back(wait_cmd);
        }
    }
    for (auto &cmd : wait_cmds) {
        cmd->Wait();
    }
}

void CudaContextRegistry::DrainAndClearContext(CUcontext ctx)
{
    if (ctx == nullptr) return;
    XDEBG("CudaContextRegistry: draining and clearing context %p", ctx);

    // 1. Wait for all commands in this context to complete
    WaitAllInContext(ctx);

    // 2. Destroy all associated queues
    auto snapshot = GetSnapshot(ctx);
    for (auto hwq_h : snapshot) {
        preempt::XQueueManager::AutoDestroy(hwq_h);
    }

    // 3. Clean up registry entry
    std::unique_lock lock(mtx_);
    registry_.erase(ctx);

    // 4. Publish a new generation so that cached context-related state
    // (e.g., per-thread default streams) can invalidate itself lazily.
    BumpDestroyGeneration();
}

} // namespace xsched::cuda
