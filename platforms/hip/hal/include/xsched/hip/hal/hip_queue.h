#pragma once

#include "xsched/types.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/hip/hal/hip.h"
#include "xsched/hip/hal/hip_command.h"
#include "xsched/hip/hal/handle.h"

namespace xsched::hip
{

class HipQueue : public preempt::HwQueue
{
public:
    HipQueue(hipStream_t stream);
    virtual ~HipQueue() = default;

    virtual void Launch(std::shared_ptr<preempt::HwCommand> hw_cmd) override;
    virtual void Synchronize() override;
    virtual void OnXQueueCreate() override;

    unsigned int          GetStreamFlags()       const    { return stream_flags_; }
    virtual XDevice       GetDevice()            override { return device_; }
    virtual HwQueueHandle GetHandle()            override { return GetHwQueueHandle(kStream); }
    virtual bool          SupportDynamicLevel()  override { return false; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelBlock; }

protected:
    const hipStream_t kStream;
    unsigned int   stream_flags_ = 0;
    XDevice        device_;
    hipCtx_t       context_ = nullptr;
    XPreemptLevel  level_ = kPreemptLevelUnknown;
};

} // namespace xsched::hip
