#include "xsched/levelzero/hal/ze_command.h"

using namespace xsched::preempt;
using namespace xsched::levelzero;

ZeListExecuteCommand::~ZeListExecuteCommand()
{
    if (following_fence_ == nullptr) return;
    ZE_ASSERT(Driver::FenceReset(following_fence_));
    fence_pool_->Push(following_fence_);
}

void ZeListExecuteCommand::Synchronize()
{
    XASSERT(following_fence_ != nullptr,
            "following_fence_ is nullptr, EnableSynchronization() should be called first");
    std::unique_lock<std::mutex> lock(fence_mtx_);
    if (fence_signaled_) return;
    ZE_ASSERT(Driver::FenceHostSynchronize(following_fence_, UINT64_MAX));
    fence_signaled_ = true;
}

bool ZeListExecuteCommand::Synchronizable()
{
    return following_fence_ != nullptr;
}

bool ZeListExecuteCommand::EnableSynchronization()
{
    if (following_fence_ != nullptr) return true;
    following_fence_ = (ze_fence_handle_t)fence_pool_->Pop();
    return following_fence_ != nullptr;
}

ze_result_t ZeListExecuteCommand::Launch(ze_command_queue_handle_t cmdq)
{
    XDEBG("ZeListExecuteCommand::Launch(cmdq: %p, cmd_list: %p, fence: %p, this: %p)",
        cmdq, kCmdList, following_fence_, this);
    return Driver::CommandQueueExecuteCommandLists(
        cmdq, 1, (ze_command_list_handle_t *)&kCmdList, following_fence_);
}

ZeKernelCommand::ZeKernelCommand(ze_command_list_handle_t hCommandList, ze_kernel_handle_t hKernel, const ze_group_count_t *pLaunchFuncArgs, ze_event_handle_t hSignalEvent, uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents)
    : HwCommand(), hCommandList_(hCommandList), hKernel_(hKernel), hSignalEvent_(hSignalEvent), numWaitEvents_(numWaitEvents)
{
    size_t launch_size = sizeof(ze_group_count_t);
    pLaunchFuncArgs_ = (ze_group_count_t *)malloc(launch_size);
    std::memcpy(pLaunchFuncArgs_, pLaunchFuncArgs, launch_size);

    size_t event_size = numWaitEvents * sizeof(ze_event_handle_t);
    phWaitEvents_ = (ze_event_handle_t *)malloc(event_size);
    std::memcpy(phWaitEvents_, phWaitEvents, event_size);

    ze_context_handle_t context;
    ZE_ASSERT(Driver::CommandListGetContextHandle(hCommandList_, &context));
    event_pool_ = EventPool::Instance(context);
}

ZeKernelCommand::~ZeKernelCommand()
{
    if (pLaunchFuncArgs_) free(pLaunchFuncArgs_);
    if (phWaitEvents_) free(phWaitEvents_);
    if (shadow_event_ != nullptr) {
        ZE_ASSERT(Driver::EventHostReset(shadow_event_));
        event_pool_->Push(shadow_event_);
    }
}

void ZeKernelCommand::Synchronize()
{
    this->WaitUntil(kCommandStateInFlight);
    if (hSignalEvent_ != nullptr) {
        ZE_ASSERT(Driver::EventHostSynchronize(hSignalEvent_, UINT64_MAX));
        return;
    }
    if (shadow_event_ != nullptr) {
        ZE_ASSERT(Driver::EventHostSynchronize(shadow_event_, UINT64_MAX));
        return;
    }
    XERRO("No event to synchronize");
}

bool ZeKernelCommand::Synchronizable()
{
    return shadow_event_ != nullptr;
}

bool ZeKernelCommand::EnableSynchronization()
{
    if (shadow_event_ != nullptr) return true;
    shadow_event_ = (ze_event_handle_t)event_pool_->Pop();
    return shadow_event_ != nullptr;
}

ze_result_t ZeKernelCommand::Launch()
{
    ZE_ASSERT(KernelArgsManager::Instance().Set(hKernel_));
    XDEBG("ZeKernelCommand::Launch(cmd_list: %p, kernel: %p, signal_event: %p, this: %p)",
        hCommandList_, hKernel_, hSignalEvent_, this);
    auto res = Driver::CommandListAppendLaunchKernel(hCommandList_, hKernel_, pLaunchFuncArgs_, hSignalEvent_, numWaitEvents_, phWaitEvents_);
    if (shadow_event_ != nullptr) ZE_ASSERT(Driver::CommandListAppendSignalEvent(hCommandList_, shadow_event_));
    return res;
}

ze_result_t ZeKernelCommand::EventQueryStatus()
{
    this->WaitUntil(kCommandStateInFlight);
    return Driver::EventQueryStatus(hSignalEvent_);
}

ze_result_t ZeKernelCommand::EventHostSignal()
{
    this->WaitUntil(kCommandStateInFlight);
    if (shadow_event_ != nullptr) ZE_ASSERT(Driver::EventHostSignal(shadow_event_));
    return Driver::EventHostSignal(hSignalEvent_);
}
