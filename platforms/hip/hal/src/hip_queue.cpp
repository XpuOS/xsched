#include "xsched/utils/pci.h"
#include "xsched/protocol/device.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/hip/hal/handle.h"
#include "xsched/hip/hal/driver.h"
#include "xsched/hip/hal/hip_queue.h"
#include "xsched/hip/hal/hip_command.h"

using namespace xsched::hip;
using namespace xsched::preempt;
using namespace xsched::protocol;

HipQueue::HipQueue(hipStream_t stream): kStream(stream)
{
    // Get hip context
    // FIXME: we should get the context from the stream
    hipCtx_t current_context = nullptr;
    HIP_ASSERT(Driver::CtxGetCurrent(&current_context));
    context_ = current_context;

    // Get device
    hipDevice_t device = 0;
    HIP_ASSERT(Driver::CtxGetDevice(&device));

    // Get device PCI address
    hipDeviceProp_t prop;
    HIP_ASSERT(Driver::GetDeviceProperties(&prop, device));
    device_ = MakeDevice(
        kDeviceTypeGPU, MakePciId(prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, 0));

    // Get stream flags
    HIP_ASSERT(Driver::StreamGetFlags(kStream, &stream_flags_));

    // Make sure no commands are running on stream_
    HIP_ASSERT(Driver::StreamSynchronize(kStream));

}

void HipQueue::Synchronize()
{
    HIP_ASSERT(Driver::StreamSynchronize(kStream));
}

void HipQueue::OnXQueueCreate()
{
    HIP_ASSERT(Driver::CtxSetCurrent(context_));
}

void HipQueue::Launch(std::shared_ptr<preempt::HwCommand> hw_cmd)
{
    auto cmd = std::dynamic_pointer_cast<HipCommand>(hw_cmd);
    XASSERT(cmd != nullptr, "hw_cmd is not a HipCommand");
    XASSERT(cmd->LaunchWrapper(kStream) == hipSuccess, "Failed to enqueue command");
}

EXPORT_C_FUNC XResult HipQueueCreate(HwQueueHandle *hwq, hipStream_t stream)
{
    if (hwq == nullptr) {
        XWARN("HipQueueCreate failed: hwq is nullptr");
        return kXSchedErrorInvalidValue;
    }
    if (stream == nullptr) {
        XWARN("HipQueueCreate failed: does not support default stream");
        return kXSchedErrorNotSupported;
    }
    
    HwQueueHandle hwq_h = GetHwQueueHandle(stream);
    auto res = HwQueueManager::Add(hwq_h, [&]() { return std::make_shared<HipQueue>(stream); });
    if (res == kXSchedSuccess) *hwq = hwq_h;
    return res;
}
