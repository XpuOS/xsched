#pragma once

#include "xsched/types.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/levelzero/hal.h"
#include "xsched/levelzero/hal/pool.h"
#include "xsched/levelzero/hal/types.h"
#include "xsched/levelzero/hal/driver.h"
#include "xsched/levelzero/hal/npu_sched.h"
#include "xsched/levelzero/hal/ze_command.h"

namespace xsched::levelzero
{

class ZeQueue : public preempt::HwQueue
{
public:
    ZeQueue(ze_device_handle_t dev, ze_command_queue_handle_t cmdq);
    virtual ~ZeQueue() { FencePool::Destroy(kCmdq); }
    
    virtual void Launch(std::shared_ptr<preempt::HwCommand> hw_cmd) override;
    virtual void Synchronize() override;

    virtual XDevice       GetDevice()            override { return device_; }
    virtual HwQueueHandle GetHandle()            override { return GetHwQueueHandle(kCmdq); }
    virtual bool          SupportDynamicLevel()  override { return false; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelBlock; }

protected:
    const ze_device_handle_t kDev;
    const ze_command_queue_handle_t kCmdq;
    XDevice device_;
};

class ZeIntelNpuQueue : public ZeQueue
{
public:
    ZeIntelNpuQueue(ze_device_handle_t dev, ze_command_queue_handle_t cmdq, ze_command_queue_priority_t prio);
    virtual ~ZeIntelNpuQueue() = default;

    virtual void Deactivate() override;
    virtual void Reactivate(const preempt::CommandLog &log) override;

    virtual bool          SupportDynamicLevel()  override { return true; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelDeactivate; }

private:
    ze_command_queue_priority_t kPrio;
};

}  // namespace xsched::levelzero
