#pragma once

#include <vector>
#include <shared_mutex>
#include <unordered_map>

#include "xsched/types.h"
#include "xsched/utils/common.h"
#include "xsched/cuda/hal/common/cuda.h"

namespace xsched::cuda
{

/// @brief Context registry that maps each CUcontext to its associated HwQueueHandles.
/// Enables context-isolated synchronization for cuCtxSynchronize and cuMemFree_v2,
/// and prevents worker thread crashes upon context destruction.
class CudaContextRegistry
{
public:
    STATIC_CLASS(CudaContextRegistry);

    /// @brief Register an HwQueueHandle to a CUDA context.
    /// Deduplicates handles automatically.
    static void Register(CUcontext ctx, HwQueueHandle hwq_h);

    /// @brief Unregister an HwQueueHandle from a CUDA context.
    /// If ctx is nullptr or handle is not found under ctx, falls back to a global scan.
    static void Unregister(CUcontext ctx, HwQueueHandle hwq_h);

    /// @brief Wait for all XQueues in the given context to complete.
    /// Uses concurrent SubmitWaitAll barriers and snapshot copying to avoid deadlocks.
    static void WaitAllInContext(CUcontext ctx);

    /// @brief Get a snapshot copy of HwQueueHandles registered under the given context.
    static std::vector<HwQueueHandle> GetSnapshot(CUcontext ctx);

    /// @brief Drain and clear all queues associated with a context upon its destruction.
    static void DrainAndClearContext(CUcontext ctx);

    /// @brief A monotonically increasing counter, incremented on every context
    /// destroy. Fast paths can cache it to detect "some context was destroyed
    /// since my cache was built" with a single atomic load.
    static uint64_t DestroyGeneration();

    /// @brief Advance the destroy generation without touching the registry.
    /// Used for destruction paths that XSched does not fully intercept
    /// (e.g., cuDevicePrimaryCtxRelease dropping the refcount to zero):
    /// cached state (per-thread default streams) will lazily revalidate
    /// itself on next use via DestroyGeneration().
    static void BumpDestroyGeneration();

private:
    static std::shared_mutex mtx_;
    static std::unordered_map<CUcontext, std::vector<HwQueueHandle>> registry_;
};

} // namespace xsched::cuda
