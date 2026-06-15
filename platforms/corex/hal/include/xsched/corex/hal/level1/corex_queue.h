#pragma once

#include "xsched/types.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/corex/hal/common/cuda.h"
#include "xsched/corex/hal/common/cudart.h"
#include "xsched/corex/hal/common/handle.h"
#include "xsched/corex/hal/common/corex_command.h"

namespace xsched::corex
{

class CorexQueueLv1 : public preempt::HwQueue
{
public:
    CorexQueueLv1(cudaStream_t stream);
    virtual ~CorexQueueLv1() = default;

    virtual void Launch(std::shared_ptr<preempt::HwCommand> hw_cmd) override;
    virtual void Synchronize() override;
    virtual void OnXQueueCreate() override;

    unsigned int          GetStreamFlags()       const    { return stream_flags_; }
    virtual XDevice       GetDevice()            override { return xdevice_; }
    virtual HwQueueHandle GetHandle()            override { return GetHwQueueHandle(kStream); }
    virtual bool          SupportDynamicLevel()  override { return false; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelBlock; }

private:
    const cudaStream_t kStream;
    unsigned int       stream_flags_ = 0;
    XDevice            xdevice_;
    int                cudevice_ = 0;
    CUcontext          context_ = nullptr;
};

} // namespace xsched::corex
