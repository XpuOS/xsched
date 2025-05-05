#pragma once

#include "xsched/types.h"
#include "xsched/levelzero/hal/ze_api.h"

namespace xsched::levelzero
{

inline HwQueueHandle GetHwQueueHandle(ze_command_queue_handle_t queue)
{
    return (HwQueueHandle)queue;
}

inline HwQueueHandle GetHwQueueHandle(ze_command_list_handle_t cmd_list)
{
    return (HwQueueHandle)cmd_list;
}

inline XDeviceType GetXDeviceType(ze_device_type_t type)
{
    switch (type)
    {
    case ZE_DEVICE_TYPE_GPU: return kDeviceTypeGPU;
    case ZE_DEVICE_TYPE_CPU: return kDeviceTypeCPU;
    case ZE_DEVICE_TYPE_FPGA: return kDeviceTypeFPGA;
    case ZE_DEVICE_TYPE_MCA: return kDeviceTypeMCA;
    case ZE_DEVICE_TYPE_VPU: return kDeviceTypeNPU; // in levelzero, VPU is just NPU
    default: return kDeviceTypeUnknown;
    }
}

} // namespace xsched::levelzero

