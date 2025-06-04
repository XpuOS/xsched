#include <map>
#include <mutex>
#include <cuxtra/cuxtra.h>

#include "xsched/cuda/hal/level3/tsg.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

using namespace xsched::cuda;

TsgContext::TsgContext(CUcontext ctx): kCtx(ctx), interrupt_count_(0)
{
    // check current context
    CUcontext current_ctx;
    CUDA_ASSERT(Driver::CtxGetCurrent(&current_ctx));
    XASSERT(current_ctx == ctx,
            "create TsgContext failed: current context (%p) does not match context (%p)",
            current_ctx, ctx);
    timeslice_ = cuXtraGetTimeslice(kCtx);
}

void TsgContext::Interrupt()
{
    size_t cnt = interrupt_count_.fetch_add(1);
    if (cnt == 0) cuXtraSetTimeslice(kCtx, 0);
}

void TsgContext::Restore()
{
    size_t cnt = interrupt_count_.fetch_sub(1);
    if (cnt == 1) cuXtraSetTimeslice(kCtx, timeslice_);
}

std::shared_ptr<TsgContext> TsgContext::Instance(CUcontext ctx)
{
    static std::mutex ctx_mtx;
    static std::map<CUcontext, std::shared_ptr<TsgContext>> ctx_map;

    std::lock_guard<std::mutex> lock(ctx_mtx);
    auto it = ctx_map.find(ctx);
    if (it != ctx_map.end()) return it->second;

    auto tsg_ctx = std::make_shared<TsgContext>(ctx);
    ctx_map[ctx] = tsg_ctx;
    return tsg_ctx;
}
