#include "xsched/protocol/device.h"
#include "xsched/cudla/hal.h"
#include "xsched/cudla/hal/driver.h"
#include "xsched/cudla/hal/cudla_queue.h"
#include "xsched/cudla/hal/cudla_assert.h"
#include "xsched/cudla/hal/cudla_command.h"

using namespace xsched::cudla;
using namespace xsched::preempt;
using namespace xsched::protocol;

CudlaQueue::CudlaQueue(cudaStream_t stream): kStream(stream)
{
    // TODO: distinguish dla device id
    device_ = MakeDevice(kDeviceTypeNPU, XDeviceId(0));
    // make sure no tasks are running on the stream
    CUDART_ASSERT(RtDriver::StreamSynchronize(kStream));
}

void CudlaQueue::Launch(std::shared_ptr<preempt::HwCommand> hw_cmd)
{
    auto cudla_command = std::dynamic_pointer_cast<CudlaCommand>(hw_cmd);
    XASSERT(cudla_command != nullptr, "hw_cmd is not a CudlaCommand");
    CUDART_ASSERT(cudla_command->LaunchWrapper(kStream));
}

void CudlaQueue::Synchronize()
{
    CUDART_ASSERT(RtDriver::StreamSynchronize(kStream));
}

EXPORT_C_FUNC XResult CudlaQueueCreate(HwQueueHandle *hwq, cudaStream_t stream)
{
    if (hwq == nullptr) {
        XWARN("CudlaQueueCreate failed: hwq is nullptr");
        return kXSchedErrorInvalidValue;
    }
    if (stream == nullptr) {
        XWARN("CudlaQueueCreate failed: stream is nullptr");
        return kXSchedErrorInvalidValue;
    }

    HwQueueHandle hwq_h = GetHwQueueHandle(stream);
    auto res = HwQueueManager::Add(hwq_h, [&]() { return std::make_shared<CudlaQueue>(stream); });
    if (res == kXSchedSuccess) *hwq = hwq_h;
    return res;
}
