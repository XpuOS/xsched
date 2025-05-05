#pragma once

#include "xsched/types.h"
#include "xsched/cuda/hal/common/cuda.h"

namespace xsched::cuda
{

inline HwQueueHandle GetHwQueueHandle(CUstream stream)
{
    return (HwQueueHandle)stream;
}

} // namespace xsched::cuda
