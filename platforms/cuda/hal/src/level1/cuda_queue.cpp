#include "xsched/utils/pci.h"
#include "xsched/utils/xassert.h"
#include "xsched/protocol/device.h"
#include "xsched/cuda/hal/common/levels.h"
#include "xsched/cuda/hal/level1/cuda_queue.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

using namespace xsched::cuda;
using namespace xsched::preempt;
using namespace xsched::protocol;

CudaQueueLv1::CudaQueueLv1(CUstream stream): kStream(stream)
{
    // get cuda context
    CUcontext stream_context = nullptr;
    CUcontext current_context = nullptr;
    CUDA_ASSERT(Driver::CtxGetCurrent(&current_context));
    CUDA_ASSERT(Driver::StreamGetCtx(stream, &stream_context));
    XASSERT(current_context == stream_context,
            "create CudaQueueLv1 failed: current context (%p) does not match stream context (%p)",
            current_context, stream_context);
    context_ = stream_context;

    // get cuda device
    CUDA_ASSERT(Driver::CtxGetDevice(&cudevice_));
    // get device PCI info
    int dom, bus, dev;
    CUDA_ASSERT(Driver::DeviceGetAttribute(&dom, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, cudevice_));
    CUDA_ASSERT(Driver::DeviceGetAttribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, cudevice_));
    CUDA_ASSERT(Driver::DeviceGetAttribute(&dev, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, cudevice_));
    xdevice_ = MakeDevice(kDeviceTypeGPU, XDeviceId(MakePciId(dom, bus, dev, 0)));

    // get stream flags
    CUDA_ASSERT(Driver::StreamGetFlags(stream, &stream_flags_));

    // make sure no commands are running on stream_
    CUDA_ASSERT(Driver::StreamSynchronize(kStream));
}

void CudaQueueLv1::Launch(std::shared_ptr<HwCommand> hw_cmd)
{
    auto cuda_cmd = std::dynamic_pointer_cast<CudaCommand>(hw_cmd);
    XASSERT(cuda_cmd != nullptr, "hw_cmd is not a CudaCommand");
    CUDA_ASSERT(cuda_cmd->LaunchWrapper(kStream));
}

void CudaQueueLv1::Synchronize()
{
    CUDA_ASSERT(Driver::StreamSynchronize(kStream));
}

void CudaQueueLv1::OnXQueueCreate()
{
    CUDA_ASSERT(Driver::CtxSetCurrent(context_));
}

CUresult CudaQueueLv1::DirectLaunch(std::shared_ptr<CudaKernelCommand> kernel, CUstream stream)
{
    return kernel->LaunchWrapper(stream);
}

EXPORT_C_FUNC XResult CudaQueueCreate(HwQueueHandle *hwq, CUstream stream)
{
    if (hwq == nullptr) {
        XWARN("CudaQueueCreate failed: hwq is nullptr");
        return kXSchedErrorInvalidValue;
    }
    if (stream == nullptr) {
        XWARN("CudaQueueCreate failed: does not support default stream");
        return kXSchedErrorNotSupported;
    }

    HwQueueHandle hwq_h = GetHwQueueHandle(stream);
    auto res = HwQueueManager::Add(hwq_h, [&]() { return xsched::cuda::CudaQueueCreate(stream); });
    if (res == kXSchedSuccess) *hwq = hwq_h;
    return res;
}
