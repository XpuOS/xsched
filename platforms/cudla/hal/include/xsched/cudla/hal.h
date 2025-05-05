#pragma once

#include "xsched/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CUstream_st *cudaStream_t;

XResult CudlaQueueCreate(HwQueueHandle *hwq, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
