#pragma once

#include <mutex>

#include "xsched/levelzero/hal.h"
#include "xsched/levelzero/hal/pool.h"
#include "xsched/levelzero/hal/driver.h"
#include "xsched/preempt/hal/hw_command.h"

namespace xsched::levelzero
{

class ZeListExecuteCommand : public preempt::HwCommand
{
public:
    ZeListExecuteCommand(ze_command_queue_handle_t cmdq, ze_command_list_handle_t cmd_list)
        : kCmdq(cmdq), kCmdList(cmd_list) { fence_pool_ = FencePool::Instance(cmdq); }
    virtual ~ZeListExecuteCommand();

    void Synchronize() override;
    bool Synchronizable() override;
    bool EnableSynchronization() override;

    ze_result_t Launch(ze_command_queue_handle_t cmdq);
    ze_command_list_handle_t GetCmdList() const { return kCmdList; }
    ze_fence_handle_t GetFollowingFence() const { return following_fence_; }

private:
    const ze_command_queue_handle_t kCmdq;
    const ze_command_list_handle_t kCmdList;

    std::mutex fence_mtx_;
    bool fence_signaled_ = false;
    ze_fence_handle_t following_fence_ = nullptr;
    std::shared_ptr<FencePool> fence_pool_ = nullptr;
};


}  // namespace xsched::levelzero
