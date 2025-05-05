
#pragma once

#include "xsched/types.h"
#include "xsched/opencl/hal.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/opencl/hal/driver.h"

namespace xsched::opencl
{

class OpenclQueue : public preempt::HwQueue
{
public:
    OpenclQueue(cl_command_queue cmdq);
    virtual ~OpenclQueue() = default;
    
    virtual void Launch(std::shared_ptr<preempt::HwCommand> hw_cmd) override;
    virtual void Synchronize() override;

    virtual XDevice       GetDevice()            override { return device_; }
    virtual HwQueueHandle GetHandle()            override { return (HwQueueHandle)cmdq_; }
    virtual bool          SupportDynamicLevel()  override { return false; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelBlock; }

private:
    cl_command_queue cmdq_;
    XDevice device_;
};

}  // namespace xsched::opencl
