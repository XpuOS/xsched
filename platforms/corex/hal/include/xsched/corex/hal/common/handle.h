#pragma once

#include "xsched/types.h"
#include "xsched/corex/hal/common/cudart.h"

namespace xsched::corex
{

inline HwQueueHandle GetHwQueueHandle(cudaStream_t stream)
{
    return (HwQueueHandle)stream;
}

} // namespace xsched::corex
