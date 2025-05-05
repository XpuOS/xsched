#include "xsched/levelzero/hal/ze_command.h"

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
