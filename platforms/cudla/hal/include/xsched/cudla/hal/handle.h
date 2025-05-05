#pragma once

#include "xsched/types.h"
#include "xsched/cudla/hal/cudart.h"

namespace xsched::cudla
{

inline HwQueueHandle GetHwQueueHandle(cudaStream_t stream)
{
    return (HwQueueHandle)stream;
}

} // namespace xsched::cudla
