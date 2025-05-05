#pragma once

#include "xsched/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CUstream_st* CUstream;

XResult CudaQueueCreate(HwQueueHandle *hwq, CUstream stream);

#ifdef __cplusplus
}
#endif
