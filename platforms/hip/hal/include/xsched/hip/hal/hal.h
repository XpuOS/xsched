#pragma once

#include "xsched/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// typedef struct hipStream_st* hipStream_t;

XResult HipQueueCreate(HwQueueHandle *hwq, hipStream_t stream);

#ifdef __cplusplus
}
#endif
