#include "xsched/cuda/hal/common/event_pool.h"

using namespace xsched::cuda;

void *ContextEventPool::Create()
{
    CUcontext current_ctx;
    CUDA_ASSERT(Driver::CtxGetCurrent(&current_ctx));
    XASSERT(current_ctx == ctx_,
            "CUDA context mismatch: using CUDA event pool of context (%p), "
            "but current context is (%p)", ctx_, current_ctx);

    CUevent event;
    CUDA_ASSERT(Driver::EventCreate(&event,
        CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING));
    return event;
}

std::mutex CudaEventPool::mutex_;
std::map<CUcontext, std::shared_ptr<ContextEventPool>> CudaEventPool::pools_;

CUevent CudaEventPool::Pop(CUcontext ctx)
{
    mutex_.lock();
    std::shared_ptr<ContextEventPool> pool = nullptr;
    auto it = pools_.find(ctx);
    if (it == pools_.end()) {
        pool = std::make_shared<ContextEventPool>(ctx);
        pools_[ctx] = pool;
    } else {
        pool = it->second;
    }
    mutex_.unlock();
    return (CUevent)pool->Pop();
}

void CudaEventPool::Push(CUcontext ctx, CUevent event)
{
    mutex_.lock();
    auto it = pools_.find(ctx);
    XASSERT(it != pools_.end(),
            "CUDA event pool not found for context (%p)", ctx);
    it->second->Push(event);
    mutex_.unlock();
}
