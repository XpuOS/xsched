#pragma once

#include "xsched/types.h"
#include "xsched/vpi/hal/vpi.h"

namespace xsched::vpi
{

inline HwQueueHandle GetHwQueueHandle(VPIStream stream)
{
    return (HwQueueHandle)stream;
}

} // namespace xsched::vpi
