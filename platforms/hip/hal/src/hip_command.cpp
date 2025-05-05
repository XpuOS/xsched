#include <cstring>
#include "xsched/hip/hal/hip.h"

#include "xsched/utils/xassert.h"
#include "xsched/types.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/hip/hal/event_pool.h"
#include "xsched/hip/hal/hip_assert.h"
#include "xsched/hip/hal/hip_command.h"
#include "xsched/hip/hal/kernel_param.h"

using namespace xsched::hip;

HipCommand::~HipCommand()
{
    if (following_event_ == nullptr) return;
    EventPool::Instance().Push(following_event_);
}

void HipCommand::Synchronize()
{
    XASSERT(following_event_ != nullptr,
            "following_event_ is nullptr, is Synchronizable() called?");
    HIP_ASSERT(Driver::EventSynchronize(following_event_));
}

bool HipCommand::Synchronizable()
{
    return following_event_ != nullptr;
}

bool HipCommand::EnableSynchronization()
{
    following_event_ = (hipEvent_t)EventPool::Instance().Pop();
    return following_event_ != nullptr;
}

hipError_t HipCommand::LaunchWrapper(hipStream_t stream)
{
    hipError_t ret = Launch(stream);
    if (UNLIKELY(ret != hipSuccess)) return ret;
    if (following_event_ != nullptr) ret = Driver::EventRecord(following_event_, stream);
    return ret;
}

HipKernelCommand::~HipKernelCommand()
{
    if (!param_copied_) return;
    if (kernel_params_ != nullptr) free(kernel_params_);
    if (param_data_ != nullptr) free(param_data_);
}

HipStaticKernelLaunchCommand::HipStaticKernelLaunchCommand(
    const void *host_func, void **params, void **extra, bool copy_param)
    : HipKernelCommand(params, extra, copy_param), host_func_(host_func)
{
    if (!copy_param) return;
    uint32_t all_params_size, num_parameters;
    KernelParamManager::Instance()->GetStaticKernelParams(host_func_, &num_parameters, &all_params_size);
    param_cnt_ = num_parameters;
    if (param_cnt_ == 0) return;
    param_copied_ = true;
    kernel_params_ = (void **)malloc(param_cnt_ * sizeof(void *)); // free in destructor
    param_data_ = (char *)malloc(all_params_size); // free in destructor
    XDEBG("HipStaticKernelLaunchCommand(%p): param_cnt_ = %lu", this, param_cnt_);
    for (size_t i = 0; i < param_cnt_; ++i) {
        size_t offset, size;
        KernelParamManager::Instance()->GetStaticKernelParamInfo(host_func_, i, &offset, &size);
        kernel_params_[i] = (void*)&param_data_[offset];
        memcpy(kernel_params_[i], original_kernel_params_[i], size);
        XDEBG("HipStaticKernelLaunchCommand(%p): param %zu, offset = %zu, size = %zu", this, i, offset, size);
    }
}

HipDynamicKernelLaunchCommand::HipDynamicKernelLaunchCommand(
    hipFunction_t function, void **kernel_params, void **extra_params, bool copy_param)
    : HipKernelCommand(kernel_params, extra_params, copy_param), function_(function)
{
    if (!copy_param) return;
    uint32_t all_params_size, num_parameters;
    KernelParamManager::Instance()->GetDynamicKernelParams(function_, &num_parameters, &all_params_size);
    XDEBG("HipDynamicKernelLaunchCommand(%p): param_cnt_ = %u, size = %u", this, num_parameters, all_params_size);
    XDEBG("HipDynamicKernelLaunchCommand(%p): kernel_params = %p, extra_params = %p", this, kernel_params, extra_params);
    
    // if the kernel_params is nullptr and extra_params is not nullptr,
    // we mofify the extra_params, instead of the kernel_params
    void** copy_src = kernel_params;
    if (copy_src == nullptr && extra_params != nullptr) {
        //  'extra' is a struct that contains the following info: {
        //   HIP_LAUNCH_PARAM_BUFFER_POINTER, kernargs,
        //   HIP_LAUNCH_PARAM_BUFFER_SIZE, &kernargs_size,
        //   HIP_LAUNCH_PARAM_END }
        copy_src = (void**) extra_params[1];
        int buffer_size = *(int*)extra_params[3];
        XDEBG("HipDynamicKernelLaunchCommand(%p): extra[1] = %p, extra[3] = %d", this, extra_params[1], *(int*)extra_params[3]);

        kernel_params_ = (void**)malloc(buffer_size);
        memcpy(kernel_params_, copy_src, buffer_size);
        param_buffer_size_ = buffer_size;
        return;
    }

    param_cnt_ = num_parameters;
    if (param_cnt_ == 0) return;
    param_copied_ = true;
    param_buffer_size_ = param_cnt_ * sizeof(void*);    
    kernel_params_ = (void **)malloc(param_buffer_size_); // free in destructor
    param_data_ = (char *)malloc(all_params_size); // free in destructor
    for (size_t i = 0; i < param_cnt_; ++i) {
        size_t offset, size;
        KernelParamManager::Instance()->GetDynamicKernelParamInfo(function_, i, &offset, &size);
        XDEBG("HipDynamicKernelLaunchCommand(%p): param %zu, size = %zu, offset = %zu", function_, i, size, offset);
        kernel_params_[i] = (void*)&param_data_[offset];
        memcpy(kernel_params_[i], copy_src[i], size);
    }
}

hipError_t HipKernelLaunchCommand::Launch(hipStream_t stream)
{
    XDEBG("HipKernelLaunchCommand(%p): host_func = %p, kernel_params = %p", this, host_func_, kernel_params_);
    return Driver::LaunchKernel(host_func_, num_blocks_, block_dim_, kernel_params_, shared_mem_bytes_, stream);
}

hipError_t HipModuleKernelLaunchCommand::Launch(hipStream_t stream)
{
    void** kernel_params = original_kernel_params_;
    void** extra_params = original_extra_params_;
    void* new_extra_params[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, kernel_params_,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, (void*)&param_buffer_size_,
        HIP_LAUNCH_PARAM_END
    };
    if (original_extra_params_ != nullptr) extra_params = new_extra_params;
    XDEBG("HipModuleKernelLaunchCommand(%p): kernel_params = %p, extra_params = %p", this, kernel_params, extra_params);
    return Driver::ModuleLaunchKernel(function_, gdx_, gdy_, gdz_, bdx_, bdy_, bdz_,
                                      shared_mem_bytes_, stream, kernel_params, extra_params);
}

hipError_t HipExtModuleKernelLaunchCommand::Launch(hipStream_t stream)
{
    void** kernel_params = original_kernel_params_;
    void** extra_params = original_extra_params_;
    void* new_extra_params[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, kernel_params_,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, (void*)&param_buffer_size_,
        HIP_LAUNCH_PARAM_END
    };
    if (original_extra_params_ != nullptr) extra_params = new_extra_params;
    XDEBG("HipExtModuleKernelLaunchCommand(%p): function = %p, kernel_params = %p, extra_params = %p", this, function_, kernel_params, extra_params);
    return Driver::ExtModuleLaunchKernel(function_, gwx_, gwy_, gwz_, lwx_, lwy_, lwz_, shm_,
                                         stream, kernel_params, extra_params, start_event_, stop_event_, flags_);
    
}

hipError_t HipMemcpyWithStreamCommand::Launch(hipStream_t stream) {
    XDEBG("HipMemcpyWithStreamCommand(%p): dst = %p, src = %p, sizeBytes = %zu, kind = %d", this, dst_, src_, sizeBytes_, (int)kind_);
    Driver::MemcpyWithStream(dst_, src_, sizeBytes_, kind_, stream);
    return Driver::StreamSynchronize(stream);
}

HipEventRecordCommand::HipEventRecordCommand(hipEvent_t event)
    : event_(event)
{
    XASSERT(event_ != nullptr, "hip event should not be nullptr");
}

HipEventRecordCommand::~HipEventRecordCommand()
{
    if (event_ == nullptr || (!destroy_event_)) return;
    HIP_ASSERT(Driver::EventDestroy(event_));
}

void HipEventWaitCommand::BeforeLaunch()
{
    if (event_record_command_) event_record_command_->Synchronize();
}

hipError_t HipEventWaitCommand::Launch(hipStream_t stream)
{
    if (!event_) return hipSuccess;
    XDEBG("HipEventWaitCommand(%p): stream = %p, event = %p, flags = %d", this, stream, event_, flags_);
    return Driver::StreamWaitEvent(stream, event_, flags_);
}
