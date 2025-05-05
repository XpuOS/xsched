
#include "xsched/protocol/device.h"
#include "xsched/opencl/hal/opencl_queue.h"
#include "xsched/opencl/hal/opencl_command.h"

using namespace xsched::opencl;

OpenclQueue::OpenclQueue(cl_command_queue cmdq): cmdq_(cmdq)
{
    // get device
    cl_device_id id;
    cl_device_type cl_type;
    XDeviceType x_type;
    Driver::GetCommandQueueInfo(cmdq_, CL_QUEUE_DEVICE, sizeof(id), &id, nullptr);
    Driver::GetDeviceInfo(id, CL_DEVICE_TYPE, sizeof(cl_type), &cl_type, nullptr);

    switch (cl_type)
    {
        case CL_DEVICE_TYPE_CPU:
            x_type = XDeviceType::kDeviceTypeCPU;
            break;
        case CL_DEVICE_TYPE_GPU:
            x_type = XDeviceType::kDeviceTypeGPU;
            break;
        default:
            x_type = XDeviceType::kDeviceTypeUnknown;
            break;
    }
    device_ = xsched::protocol::MakeDevice(x_type, 0);
}

void OpenclQueue::Launch(std::shared_ptr<preempt::HwCommand> hw_cmd)
{
    auto cmd = std::dynamic_pointer_cast<OpenclCommand>(hw_cmd);
    if (cmd == nullptr) return;
    cmd->Launch(cmdq_);
}

void OpenclQueue::Synchronize()
{
    Driver::Finish(cmdq_);
}

extern "C" XResult OpenclQueueCreate(HwQueueHandle *hwq, cl_command_queue cmdq)
{
    HwQueueHandle hwq_h = (HwQueueHandle)cmdq;
    auto res = xsched::preempt::HwQueueManager::Add(hwq_h, [&]() {
        return std::make_shared<OpenclQueue>(cmdq);
    });
    if (res == kXSchedSuccess) *hwq = hwq_h;
    return res;
}
