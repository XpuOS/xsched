#include "xsched/utils/pci.h"
#include "xsched/protocol/device.h"
#include "xsched/levelzero/hal/ze_queue.h"
#include "xsched/levelzero/hal/ze_assert.h"

using namespace xsched::preempt;
using namespace xsched::protocol;
using namespace xsched::levelzero;

ZeQueue::ZeQueue(ze_device_handle_t dev, ze_command_queue_handle_t cmdq): kDev(dev), kCmdq(cmdq)
{
    ze_device_properties_t dev_props;
    ze_pci_ext_properties_t pci_ext_props;
    dev_props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    dev_props.pNext = nullptr;
    pci_ext_props.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
    pci_ext_props.pNext = nullptr;
    ZE_ASSERT(Driver::DeviceGetProperties(kDev, &dev_props));
    ZE_ASSERT(Driver::DevicePciGetPropertiesExt(kDev, &pci_ext_props));

    XDeviceId id = MakePciId(pci_ext_props.address.domain, pci_ext_props.address.bus,
                             pci_ext_props.address.device, pci_ext_props.address.function);
    device_ = MakeDevice(GetXDeviceType(dev_props.type), id);
    ZE_ASSERT(Driver::CommandQueueSynchronize(kCmdq, UINT64_MAX));
}

void ZeQueue::Launch(std::shared_ptr<preempt::HwCommand> hw_cmd)
{
    auto cmd = std::dynamic_pointer_cast<ZeListExecuteCommand>(hw_cmd);
    XASSERT(cmd != nullptr, "hw_cmd is not a ZeListExecuteCommand");
    ZE_ASSERT(cmd->Launch(kCmdq));
}

void ZeQueue::Synchronize()
{
    ZE_ASSERT(Driver::CommandQueueSynchronize(kCmdq, UINT64_MAX));
}

ZeIntelNpuQueue::ZeIntelNpuQueue(ze_device_handle_t dev, ze_command_queue_handle_t cmdq, ze_command_queue_priority_t prio)
: ZeQueue(dev, cmdq), kPrio(prio)
{
    
}

void ZeIntelNpuQueue::Deactivate()
{
    // current driver uses one hardware command queue for ze_command_queue_handle_t(s) of one priority
    npu_sched_suspend_cmdq(kPrio);
}

void ZeIntelNpuQueue::Reactivate(const preempt::CommandLog &)
{
    npu_sched_resume_cmdq(kPrio);
}

EXPORT_C_FUNC XResult ZeQueueCreate(HwQueueHandle *hwq, ze_device_handle_t dev, ze_command_queue_handle_t cmdq)
{
    if (hwq == nullptr) {
        XWARN("ZeQueueCreate failed: hwq is nullptr");
        return kXSchedErrorInvalidValue;
    }
    if (dev == nullptr || cmdq == nullptr) {
        XWARN("ZeQueueCreate failed: dev or cmdq is nullptr");
        return kXSchedErrorInvalidValue;
    }

    ze_device_properties_t dev_props;
    dev_props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    dev_props.pNext = nullptr;
    ZE_ASSERT(Driver::DeviceGetProperties(dev, &dev_props));
    
    HwQueueHandle hwq_h = GetHwQueueHandle(cmdq);
    auto res = HwQueueManager::Add(hwq_h, [&]() -> std::shared_ptr<ZeQueue> {
        if (dev_props.type == ZE_DEVICE_TYPE_VPU) {
            return std::make_shared<ZeIntelNpuQueue>(dev, cmdq, ZE_COMMAND_QUEUE_PRIORITY_NORMAL);
        }
        return std::make_shared<ZeQueue>(dev, cmdq);
    });
    if (res == kXSchedSuccess) *hwq = hwq_h;
    return res;
}
