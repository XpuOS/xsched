#pragma once

#include "xsched/types.h"
#include "xsched/opencl/hal/types.h"
#include "xsched/opencl/hal/driver.h"
#include "xsched/preempt/hal/hw_queue.h"

namespace xsched::opencl
{

class OclQueue : public preempt::HwQueue
{
public:
    OclQueue(cl_command_queue cmdq);
    virtual ~OclQueue() = default;
    
    virtual void Launch(std::shared_ptr<preempt::HwCommand> hw_cmd) override;
    virtual void Synchronize() override;

    virtual XDevice       GetDevice()            override { return device_; }
    virtual HwQueueHandle GetHandle()            override { return GetHwQueueHandle(kCmdq); }
    virtual bool          SupportDynamicLevel()  override { return false; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelBlock; }

private:
    const cl_command_queue kCmdq;
    XDevice device_;
};

} // namespace xsched::opencl
