#pragma once

#include <mutex>

#include "xsched/levelzero/hal.h"
#include "xsched/levelzero/hal/pool.h"
#include "xsched/levelzero/hal/driver.h"
#include "xsched/levelzero/hal/ze_kernel_arg.h"
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

class ZeKernelCommand : public preempt::HwCommand
{
public:
    ZeKernelCommand(ze_command_list_handle_t hCommandList, ze_kernel_handle_t hKernel, const ze_group_count_t *pLaunchFuncArgs, ze_event_handle_t hSignalEvent, uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);
    ~ZeKernelCommand();

    void Synchronize() override;
    bool Synchronizable() override;
    bool EnableSynchronization() override;

    ze_result_t Launch();
    ze_result_t EventHostSignal();
    ze_result_t EventQueryStatus();

private:
    const ze_command_list_handle_t hCommandList_;
    const ze_kernel_handle_t hKernel_;
    ze_group_count_t *pLaunchFuncArgs_;
    const ze_event_handle_t hSignalEvent_;
    const uint32_t numWaitEvents_;
    ze_event_handle_t *phWaitEvents_;

    ze_event_handle_t shadow_event_ = nullptr;
    std::shared_ptr<EventPool> event_pool_ = nullptr;
};

} // namespace xsched::levelzero
