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
    // Get hip context — DTK may not support hipCtxGetCurrent (error 201),
    // so fall back gracefully; context_=nullptr means CtxSetCurrent is a no-op.
    hipCtx_t current_context = nullptr;
    if (Driver::CtxGetCurrent(&current_context) != hipSuccess) {
        current_context = nullptr;
    }
    context_ = current_context;

    // Get device from stream — hipStreamGetDevice queries the stream object
    // directly and works on both ROCm and DTK without requiring a current context.
    hipDevice_t device = 0;
    if (Driver::StreamGetDevice(kStream, &device) != hipSuccess) {
        int ordinal = 0;
        HIP_ASSERT(Driver::GetDevice(&ordinal));
        HIP_ASSERT(Driver::DeviceGet(&device, ordinal));
    }

    // Build device ID from UUID instead of hipGetDeviceProperties, because
    // DTK does not export hipGetDeviceProperties (neither base nor R0600 variant).
    // hipDeviceGetUuid is available on both ROCm and DTK.
    hipUUID uuid;
    HIP_ASSERT(Driver::DeviceGetUuid(&uuid, device));
    uint32_t device_id = 0;
    for (int i = 0; i < 16; i++) {
        device_id ^= static_cast<uint32_t>(static_cast<unsigned char>(uuid.bytes[i]))
                     << ((i % 4) * 8);
    }
    device_ = MakeDevice(kDeviceTypeGPU, device_id);

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
    // context_ may be nullptr on platforms that don't support
    // hipCtxGetCurrent (e.g., Hygon DTK), skip context switching.
    if (context_ != nullptr) {
        HIP_ASSERT(Driver::CtxSetCurrent(context_));
    }
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
