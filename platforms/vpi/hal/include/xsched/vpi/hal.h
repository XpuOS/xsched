#pragma once

#include "xsched/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VPIStreamImpl *VPIStream;

XResult VpiQueueCreate(HwQueueHandle *hwq, VPIStream stream);

#ifdef __cplusplus
}
#endif
