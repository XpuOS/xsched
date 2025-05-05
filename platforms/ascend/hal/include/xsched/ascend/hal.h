#pragma once

#include "xsched/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *aclrtStream;

XResult AclQueueCreate(HwQueueHandle *hwq, aclrtStream stream);

#ifdef __cplusplus
}
#endif
