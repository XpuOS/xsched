#include "xsched/utils/pci.h"
#include "xsched/protocol/device.h"
#include "xsched/opencl/hal.h"
#include "xsched/opencl/hal/ocl_queue.h"
#include "xsched/opencl/hal/ocl_assert.h"
#include "xsched/opencl/hal/ocl_command.h"

using namespace xsched::opencl;
using namespace xsched::preempt;
using namespace xsched::protocol;

OclQueue::OclQueue(cl_command_queue cmdq): kCmdq(cmdq)
{
    // get device
    cl_device_id id;
    cl_device_type type;
    cl_device_pci_bus_info_khr pci;
    OCL_ASSERT(Driver::GetCommandQueueInfo(kCmdq, CL_QUEUE_DEVICE, sizeof(id), &id, nullptr));
    OCL_ASSERT(Driver::GetDeviceInfo(id, CL_DEVICE_TYPE, sizeof(type), &type, nullptr));
    OCL_ASSERT(Driver::GetDeviceInfo(id, CL_DEVICE_PCI_BUS_INFO_KHR, sizeof(pci), &pci, nullptr));
    device_ = MakeDevice(GetXDeviceType(type), XDeviceId(MakePciId(
        pci.pci_domain, pci.pci_bus, pci.pci_device, pci.pci_function)));

    // make sure no tasks are running on stream_
    XDEBG("OclQueue (%p) created for cmdq (%p)", this, kCmdq);
    OCL_ASSERT(Driver::Flush(kCmdq));
    OCL_ASSERT(Driver::Finish(kCmdq));
}

void OclQueue::Launch(std::shared_ptr<preempt::HwCommand> hw_cmd)
{
    auto ocl_cmd = std::dynamic_pointer_cast<OclCommand>(hw_cmd);
    XASSERT(ocl_cmd != nullptr, "hw_cmd is not an OclCommand");
    OCL_ASSERT(ocl_cmd->LaunchWrapper(kCmdq));
}

void OclQueue::Synchronize()
{
    XDEBG("synchronize cmdq (%p)", kCmdq);
    OCL_ASSERT(Driver::Flush(kCmdq));
    OCL_ASSERT(Driver::Finish(kCmdq));
}

EXPORT_C_FUNC XResult OclQueueCreate(HwQueueHandle *hwq, cl_command_queue cmdq)
{
    if (hwq == nullptr) {
        XWARN("OclQueueCreate failed: hwq is nullptr");
        return kXSchedErrorInvalidValue;
    }
    if (cmdq == nullptr) {
        XWARN("OclQueueCreate failed: cmdq is nullptr");
        return kXSchedErrorInvalidValue;
    }

    HwQueueHandle hwq_h = GetHwQueueHandle(cmdq);
    auto res = HwQueueManager::Add(hwq_h, [&]() { return std::make_shared<OclQueue>(cmdq); });
    if (res == kXSchedSuccess) *hwq = hwq_h;
    return res;
}
