#pragma once

#include "xsched/types.h"
#include "xsched/ascend/hal/acl.h"

namespace xsched::ascend
{

inline HwQueueHandle GetHwQueueHandle(aclrtStream stream)
{
    return (HwQueueHandle)stream;
}

} // namespace xsched::ascend
