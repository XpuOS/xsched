#pragma once

#include "xsched/types.h"
#include "xsched/vpi/hal/vpi.h"
#include "xsched/vpi/hal/handle.h"
#include "xsched/preempt/hal/hw_queue.h"

namespace xsched::vpi
{

class VpiQueue : public preempt::HwQueue
{
public:
    VpiQueue(VPIStream stream);
    virtual ~VpiQueue() = default;

    virtual void Launch(std::shared_ptr<preempt::HwCommand> hw_cmd) override;
    virtual void Synchronize() override;

    virtual XDevice       GetDevice()            override { return device_; }
    virtual HwQueueHandle GetHandle()            override { return GetHwQueueHandle(kStream); }
    virtual bool          SupportDynamicLevel()  override { return false; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelBlock; }

private:
    const VPIStream kStream;
    XDevice device_;
};

} // namespace xsched::vpi
