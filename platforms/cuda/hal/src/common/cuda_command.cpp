#include <cstring>
#include <cuxtra/cuxtra.h>

#include "xsched/types.h"
#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/common/event_pool.h"
#include "xsched/cuda/hal/common/cuda_assert.h"
#include "xsched/cuda/hal/common/cuda_command.h"

using namespace xsched::cuda;

CudaCommand::CudaCommand(preempt::XCommandProperties props): HwCommand(props)
{
    CUDA_ASSERT(Driver::CtxGetCurrent(&ctx_));
    XASSERT(ctx_ != nullptr, "current context of the calling thread is nullptr");
}

CudaCommand::~CudaCommand()
{
    if (following_event_ == nullptr) return;
    CudaEventPool::Push(ctx_, following_event_);
}

void CudaCommand::Synchronize()
{
    XASSERT(following_event_ != nullptr,
            "following_event_ is nullptr, EnableSynchronization() should be called first");
    CUDA_ASSERT(Driver::EventSynchronize(following_event_));
}

bool CudaCommand::Synchronizable()
{
    return following_event_ != nullptr;
}

bool CudaCommand::EnableSynchronization()
{
    following_event_ = CudaEventPool::Pop(ctx_);
    return following_event_ != nullptr;
}

CUresult CudaCommand::LaunchWrapper(CUstream stream)
{
    CUresult ret = Launch(stream);
    if (UNLIKELY(ret != CUDA_SUCCESS)) return ret;
    if (following_event_ != nullptr) ret = Driver::EventRecord(following_event_, stream);
    return ret;
}

CudaKernelCommand::CudaKernelCommand(CUfunction func, void **params, void **extra, bool deep_copy)
    : CudaCommand(preempt::kCommandPropertyDeactivatable)
    , kFunc(func), params_(params), extra_(extra)
{
    if (!deep_copy) return;

    param_cnt_ = cuXtraGetParamCount(kFunc);
    if (param_cnt_ == 0) return;
    if (params == nullptr) {
        XWARN("kernel_params of %p is nullptr", func);
        return;
    }

    deep_copy_ = true;
    params_ = (void **)malloc(param_cnt_ * sizeof(void *));
    // Allocate a continuous buffer for all of the params
    // buffer size = last param offset + last param size
    size_t last_offset, last_size;
    cuXtraGetParamInfo(kFunc, param_cnt_ - 1, &last_offset, &last_size, nullptr);
    size_t buffer_size = last_offset + last_size;
    param_data_ = (char *)malloc(buffer_size);

    for (size_t i = 0; i < param_cnt_; ++i) {
        size_t offset, size;
        cuXtraGetParamInfo(kFunc, i, &offset, &size, nullptr);
        params_[i] = (void*)&param_data_[offset];
        memcpy(params_[i], params[i], size);
    }
}

CudaKernelCommand::~CudaKernelCommand()
{
    if (!deep_copy_) return;
    free(param_data_);
    free(params_);
}

CudaKernelLaunchExCommand::CudaKernelLaunchExCommand(const CUlaunchConfig *cfg, CUfunction func,
                                                     void **params, void **extra, bool deep_copy)
    : CudaKernelCommand(func, params, extra, deep_copy)
{
    if (cfg == nullptr) {
        XWARN("CUlaunchConfig of %p is nullptr", func);
        return;
    }

    memcpy(&cfg_, cfg, sizeof(CUlaunchConfig));
    if (cfg->attrs == nullptr || !deep_copy_) return;
    cfg_.attrs = (CUlaunchAttribute *)malloc(cfg->numAttrs * sizeof(CUlaunchAttribute));
    memcpy(cfg_.attrs, cfg->attrs, cfg->numAttrs * sizeof(CUlaunchAttribute));
}

CudaKernelLaunchExCommand::~CudaKernelLaunchExCommand()
{
    if (cfg_.attrs == nullptr || !deep_copy_) return;
    free(cfg_.attrs);
}

CUresult CudaKernelLaunchExCommand::Launch(CUstream stream)
{
    cfg_.hStream = stream;
    return Driver::LaunchKernelEx(&cfg_, kFunc, params_, extra_);
}

CudaEventRecordCommand::CudaEventRecordCommand(CUevent event)
    : CudaCommand(preempt::kCommandPropertyIdempotent), event_(event)
{
    XASSERT(event_ != nullptr, "cuda event should not be nullptr");
}

CudaEventRecordCommand::~CudaEventRecordCommand()
{
    if (event_ == nullptr || (!destroy_event_)) return;
    CUDA_ASSERT(Driver::EventDestroy(event_));
}

void CudaEventWaitCommand::BeforeLaunch()
{
    if (event_cmd_) event_cmd_->Wait(); // recorded on an XQueue, wait it on the XQueue
}

CUresult CudaEventWaitCommand::Launch(CUstream stream)
{
    if (!event_) return CUDA_SUCCESS; // already waited in BeforeLaunch()
    return Driver::StreamWaitEvent(stream, event_, flags_);
}
