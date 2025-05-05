#include "xsched/protocol/device.h"
#include "xsched/ascend/hal.h"
#include "xsched/ascend/hal/driver.h"
#include "xsched/ascend/hal/acl_queue.h"
#include "xsched/ascend/hal/acl_assert.h"
#include "xsched/ascend/hal/acl_command.h"

using namespace xsched::ascend;
using namespace xsched::preempt;
using namespace xsched::protocol;

AclQueue::AclQueue(aclrtStream stream): kStream(stream)
{
    ACL_ASSERT(Driver::rtGetCurrentContext(&context_));
    ACL_ASSERT(Driver::rtGetDevice(&device_id_));
    device_ = MakeDevice(kDeviceTypeNPU, XDeviceId(device_id_));
}

void AclQueue::Launch(std::shared_ptr<preempt::HwCommand> hw_cmd)
{
    auto acl_cmd = std::dynamic_pointer_cast<AclCommand>(hw_cmd);
    XASSERT(acl_cmd != nullptr, "hw_cmd is not an AclCommand");
    ACL_ASSERT(acl_cmd->LaunchWrapper(kStream));
}

void AclQueue::Synchronize()
{
    aclrtContext cur_ctx = nullptr;
    ACL_ASSERT(Driver::rtGetCurrentContext(&cur_ctx));
    if (cur_ctx != context_) {
        XWARN("stream context (%p) != current context (%p), override current", context_, cur_ctx);
        ACL_ASSERT(Driver::rtSetCurrentContext(context_));
    }
    ACL_ASSERT(Driver::rtSynchronizeStream(kStream));
}

void AclQueue::OnXQueueCreate()
{
    ACL_ASSERT(Driver::rtSetCurrentContext(context_));
}

EXPORT_C_FUNC XResult AclQueueCreate(HwQueueHandle *hwq, aclrtStream stream)
{
    if (hwq == nullptr) {
        XWARN("AclQueueCreate failed: hwq is nullptr");
        return kXSchedErrorInvalidValue;
    }
    if (stream == nullptr) {
        XWARN("AclQueueCreate failed: stream is nullptr");
        return kXSchedErrorInvalidValue;
    }

    HwQueueHandle hwq_h = GetHwQueueHandle(stream);
    auto res = HwQueueManager::Add(hwq_h, [&]() { return std::make_shared<AclQueue>(stream); });
    if (res == kXSchedSuccess) *hwq = hwq_h;
    return res;
}
