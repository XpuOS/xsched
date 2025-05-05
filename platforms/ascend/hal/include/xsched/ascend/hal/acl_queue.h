#pragma once

#include "xsched/types.h"
#include "xsched/ascend/hal/acl.h"
#include "xsched/ascend/hal/handle.h"
#include "xsched/preempt/hal/hw_queue.h"

namespace xsched::ascend
{

class AclQueue : public preempt::HwQueue
{
public:
    AclQueue(aclrtStream stream);
    virtual ~AclQueue() = default;

    virtual void Launch(std::shared_ptr<preempt::HwCommand> hw_cmd) override;
    virtual void Synchronize() override;
    virtual void OnXQueueCreate() override;

    virtual XDevice       GetDevice()            override { return device_; }
    virtual HwQueueHandle GetHandle()            override { return GetHwQueueHandle(kStream); }
    virtual bool          SupportDynamicLevel()  override { return false; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelBlock; }

private:
    const aclrtStream kStream;
    XDevice           device_;
    int32_t           device_id_ = 0;
    aclrtContext      context_ = nullptr;
};

} // namespace xsched::ascend
