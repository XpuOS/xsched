#pragma once

#include "xsched/types.h"
#include "xsched/hip/hal/hip.h"

namespace xsched::hip
{

inline HwQueueHandle GetHwQueueHandle(hipStream_t stream)
{
    return (HwQueueHandle)stream;
}

} // namespace xsched::hip
