#pragma once

#include "xsched/types.h"
#include "xsched/cudla/hal/cudla.h"
#include "xsched/cudla/hal/cudart.h"
#include "xsched/cudla/hal/handle.h"
#include "xsched/preempt/hal/hw_queue.h"

namespace xsched::cudla
{

class CudlaQueue : public preempt::HwQueue
{
public:
    CudlaQueue(cudaStream_t stream);
    virtual ~CudlaQueue() = default;

    virtual void Launch(std::shared_ptr<preempt::HwCommand> hw_cmd) override;
    virtual void Synchronize() override;

    virtual XDevice       GetDevice()            override { return device_; }
    virtual HwQueueHandle GetHandle()            override { return GetHwQueueHandle(kStream); }
    virtual bool          SupportDynamicLevel()  override { return false; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelBlock; }

private:
    const cudaStream_t kStream;
    XDevice device_;
};

} // namespace xsched::cudla
