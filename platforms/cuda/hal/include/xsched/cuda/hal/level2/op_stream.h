#pragma once

#include <map>
#include <mutex>

#include "xsched/utils/common.h"
#include "xsched/cuda/hal/common/cuda.h"

namespace xsched::cuda
{

class OpStreamManager
{
public:
    STATIC_CLASS(OpStreamManager);
    static CUstream GetOpStream(CUcontext context);
};

} // namespace xsched::cuda
