#include <cstring>

#include "xsched/utils/log.h"
#include "xsched/utils/xassert.h"
#include "xsched/corex/hal/common/event_pool.h"
#include "xsched/corex/hal/common/corex_assert.h"
#include "xsched/corex/hal/common/corex_command.h"

using namespace xsched::corex;
using namespace xsched::preempt;

CorexCommand::~CorexCommand()
{
    if (following_event_ == nullptr) return;
    EventPool::Instance().Push(following_event_);
}

void CorexCommand::Synchronize()
{
    XASSERT(following_event_ != nullptr,
            "following_event_ is nullptr, EnableSynchronization() should be called first");
    COREXRT_ASSERT(RtDriver::EventSynchronize(following_event_));
}

bool CorexCommand::Synchronizable()
{
    return following_event_ != nullptr;
}

bool CorexCommand::EnableSynchronization()
{
    following_event_ = (cudaEvent_t)EventPool::Instance().Pop();
    return following_event_ != nullptr;
}

cudaError_t CorexCommand::LaunchWrapper(cudaStream_t stream)
{
    cudaError_t ret = Launch(stream);
    if (UNLIKELY(ret != cudaSuccess)) return ret;
    if (following_event_ != nullptr) ret = RtDriver::EventRecord(following_event_, stream);
    return ret;
}

CorexKernelCommand::CorexKernelCommand(const void *func, void **params, bool deep_copy)
    : CorexCommand(kCommandPropertyDeactivatable), func_(func), params_(params)
{
    if (!deep_copy) return;
    if (params == nullptr) {
        XWARN("kernel_args of %p is nullptr", func);
        return;
    }

    cudaKernel_t kernel = nullptr;
    cudaError_t res = RtDriver::GetKernel(&kernel, func);
    if (res != cudaSuccess) RtDriver::GetLastError();
    XASSERT(kernel != nullptr, "fail to get kernel for func %p", func);

    std::vector<size_t> offsets;
    std::vector<size_t> sizes;

    param_cnt_ = 0;
    while (true) {
        size_t offset = 0, size = 0;
        CUresult res = Driver::KernelGetParamInfo(CUkernel(kernel), param_cnt_, &offset, &size);
        if (res != CUDA_SUCCESS || size == 0) break;
        offsets.push_back(offset);
        sizes.push_back(size);
        ++param_cnt_;
    }

    if (param_cnt_ == 0) {
        XWARN("kernel %p has no params", func);
        return;
    }

    deep_copy_ = true;
    params_ = (void **)malloc(param_cnt_ * sizeof(void *));
    // Allocate a continuous buffer for all of the params
    // buffer size = last param offset + last param size
    size_t buffer_size = offsets.back() + sizes.back();
    param_data_ = (char *)malloc(buffer_size);

    for (size_t i = 0; i < param_cnt_; ++i) {
        size_t offset = offsets[i];
        size_t size = sizes[i];
        params_[i] = (void*)&param_data_[offset];
        memcpy(params_[i], params[i], size);
    }
}

CorexKernelCommand::~CorexKernelCommand()
{
    if (!deep_copy_) return;
    free(param_data_);
    free(params_);
}

CorexMemcpyCommand::CorexMemcpyCommand(void *dst, const void *src, size_t size,
                                       cudaMemcpyKind kind, bool ptsz)
    : dst_(dst), src_(src), size_(size), kind_(kind), ptsz_(ptsz)
{
    if (kind != cudaMemcpyHostToDevice) return;

    // save src if src is not pinned memory
    cudaPointerAttributes attr;
    cudaError_t err = RtDriver::PointerGetAttributes(&attr, src);
    if (err == cudaSuccess && attr.type == cudaMemoryTypeHost) return;

    COREXRT_ASSERT(RtDriver::MallocHost(&pinned_src_, size));
    memcpy(pinned_src_, src, size);
    src_ = pinned_src_;
}

CorexMemcpyCommand::~CorexMemcpyCommand()
{
    if (pinned_src_ == nullptr) return;
    COREXRT_ASSERT(RtDriver::FreeHost(pinned_src_));
}

CorexEventRecordCommand::CorexEventRecordCommand(cudaEvent_t event, bool ptsz)
    : CorexCommand(kCommandPropertyIdempotent), event_(event), ptsz_(ptsz)
{
    XASSERT(event_ != nullptr, "cuda event should not be nullptr");
}

CorexEventRecordCommand::~CorexEventRecordCommand()
{
    if (event_ == nullptr || !destroy_event_) return;
    COREXRT_ASSERT(RtDriver::EventDestroy(event_));
}

void CorexEventWaitCommand::BeforeLaunch()
{
    if (event_cmd_ != nullptr) event_cmd_->Wait(); // recorded on an XQueue, wait it on the XQueue
}

cudaError_t CorexEventWaitCommand::Launch(cudaStream_t stream)
{
    if (event_ == nullptr) return cudaSuccess; // already waited in BeforeLaunch()
    return COREXRT_LAUNCH(ptsz_, StreamWaitEvent, stream, event_, flags_);
}
