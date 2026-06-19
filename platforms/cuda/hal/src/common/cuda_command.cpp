#include <cstring>
#include <cuxtra/cuxtra.h>

#include "xsched/utils/log.h"
#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/common/options.h"
#include "xsched/cuda/hal/common/event_pool.h"
#include "xsched/cuda/hal/common/cuda_assert.h"
#include "xsched/cuda/hal/common/cuda_command.h"

using namespace xsched::cuda;

CudaCommand::CudaCommand(preempt::XCommandProperties props)
    : HwCommand(GetCudaLv3Implementation() == kCudaLv3ImplementationTsg
                ? props | preempt::kCommandPropertyDeactivatable : props)
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
    if (following_event_ != nullptr) return true;
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
    // cuXtraKernelGetFunction() will convert func to CUfunction if func is a CUkernel.
    , kFunc(func), kFuncHandle(cuXtraKernelGetFunction((CUkernel)func))
    , params_(params), extra_(extra), param_cnt_(cuXtraGetParamCount(kFuncHandle))
{
    if (!deep_copy) return;

    /// @FIXME: even if param_cnt_ == 0, params or extra can be non-nullptr.
    if (param_cnt_ == 0) return;
    if (params == nullptr && extra == nullptr) {
        XWARN("params and extra of kernel (%p) are both nullptr", func);
        return;
    }
    if (params != nullptr && extra != nullptr) {
        XWARN("illegally using both params (%p) and extra (%p) of kernel %p", params, extra, func);
        return;
    }

    deep_copy_ = true;
    params_ = (void **)malloc(param_cnt_ * sizeof(void *));

    if (params != nullptr) {
        // Allocate a continuous buffer for all of the params
        // buffer size = last param offset + last param size
        size_t last_offset, last_size;
        cuXtraGetParamInfo(kFuncHandle, param_cnt_ - 1, &last_offset, &last_size, nullptr);
        size_t buffer_size = last_offset + last_size;
        param_data_ = (char *)malloc(buffer_size);

        for (size_t i = 0; i < param_cnt_; ++i) {
            size_t offset, size;
            cuXtraGetParamInfo(kFuncHandle, i, &offset, &size, nullptr);
            params_[i] = (void*)&param_data_[offset];
            memcpy(params_[i], params[i], size);
        }
    } else if (extra != nullptr) {
        // Get extra buffer and its size from extra map.
        void *extra_buffer = nullptr;
        cuXtraGetExtraBuffer(extra, &extra_buffer, &extra_buffer_size_);
        // We have checked param_cnt_ > 0.
        XASSERT(extra_buffer != nullptr && extra_buffer_size_ > 0,
                "invalid extra buffer (%p) and size (%zu)", extra_buffer, extra_buffer_size_);

        // Deep-copy extra buffer to extra_data_.
        extra_data_ = (char *)malloc(extra_buffer_size_);
        memcpy(extra_data_, extra_buffer, extra_buffer_size_);

        // Set extra map.
        extra_ = (void **)malloc(5 * sizeof(void *));
        extra_[0] = CU_LAUNCH_PARAM_BUFFER_POINTER;
        extra_[1] = extra_data_;
        extra_[2] = CU_LAUNCH_PARAM_BUFFER_SIZE;
        extra_[3] = (void *)&extra_buffer_size_;
        extra_[4] = CU_LAUNCH_PARAM_END;

        // Set params_ to point to extra_data_.
        for (size_t i = 0; i < param_cnt_; ++i) {
            size_t offset, size;
            cuXtraGetParamInfo(kFuncHandle, i, &offset, &size, nullptr);
            XASSERT(offset + size <= extra_buffer_size_,
                    "extra[%zu] out of range: offset (%zu) + size (%zu) > buffer size (%zu)",
                    i, offset, size, extra_buffer_size_);
            params_[i] = (void*)&extra_data_[offset];
        }
    }
}

CudaKernelCommand::~CudaKernelCommand()
{
    if (!deep_copy_) return;
    if (param_data_) free(param_data_);
    if (extra_data_) free(extra_data_);
    if (params_) free(params_);
    if (extra_) free(extra_);
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
    void **params = params_;
    if (deep_copy_ && extra_ != nullptr) params = nullptr;
    return Driver::LaunchKernelEx(&cfg_, kFunc, params, extra_);
}

CudaEventRecordCommand::CudaEventRecordCommand(CUevent event)
    : CudaCommand(preempt::kCommandPropertyIdempotent), event_(event)
{
    XASSERT(event_ != nullptr, "cuda event should not be nullptr");
}

CudaEventRecordCommand::~CudaEventRecordCommand()
{
    if (event_ == nullptr || (!destroy_event_)) return;
    CUDA_ASSERT(Driver::EventDestroy_v2(event_));
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
