#pragma once

#include "xsched/types.h"
#include "xsched/opencl/hal/cl.h"

namespace xsched::opencl
{

inline HwQueueHandle GetHwQueueHandle(cl_command_queue cmdq)
{
    return (HwQueueHandle)cmdq;
}

inline XDeviceType GetXDeviceType(cl_device_type type)
{
    switch (type)
    {
    case CL_DEVICE_TYPE_CPU: return kDeviceTypeCPU;
    case CL_DEVICE_TYPE_GPU: return kDeviceTypeGPU;
    case CL_DEVICE_TYPE_ACCELERATOR: return kDeviceTypeFPGA; // TODO: other types
    default: return kDeviceTypeUnknown;
    }
}

} // namespace xsched::opencl
