#include "xsched/protocol/device.h"
#include "xsched/vpi/hal.h"
#include "xsched/vpi/hal/driver.h"
#include "xsched/vpi/hal/vpi_queue.h"
#include "xsched/vpi/hal/vpi_assert.h"
#include "xsched/vpi/hal/vpi_command.h"

using namespace xsched::vpi;
using namespace xsched::preempt;
using namespace xsched::protocol;

VpiQueue::VpiQueue(VPIStream stream): kStream(stream)
{
    // TODO: distinguish device type and id
    device_ = MakeDevice(kDeviceTypeASIC, XDeviceId(0));
    // make sure no tasks are running on the stream
    VPI_ASSERT(Driver::StreamSync(kStream));
}

void VpiQueue::Launch(std::shared_ptr<preempt::HwCommand> hw_cmd)
{
    auto vpi_command = std::dynamic_pointer_cast<VpiCommand>(hw_cmd);
    XASSERT(vpi_command != nullptr, "hw_cmd is not a VpiCommand");
    VPI_ASSERT(vpi_command->LaunchWrapper(kStream));
}

void VpiQueue::Synchronize()
{
    VPI_ASSERT(Driver::StreamSync(kStream));
}

EXPORT_C_FUNC XResult VpiQueueCreate(HwQueueHandle *hwq, VPIStream stream)
{
    if (hwq == nullptr) {
        XWARN("VpiQueueCreate failed: hwq is nullptr");
        return kXSchedErrorInvalidValue;
    }
    if (stream == nullptr) {
        XWARN("VpiQueueCreate failed: stream is nullptr");
        return kXSchedErrorInvalidValue;
    }

    HwQueueHandle hwq_h = GetHwQueueHandle(stream);
    auto res = HwQueueManager::Add(hwq_h, [&]() { return std::make_shared<VpiQueue>(stream); });
    if (res == kXSchedSuccess) *hwq = hwq_h;
    return res;
}
