#pragma once

#include "xsched/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CUstream_st* CUstream;
typedef struct CUctx_st* CUcontext;

XResult CudaQueueCreate(HwQueueHandle *hwq, CUstream stream);

XResult CudaQueueGet(HwQueueHandle *hwq, CUstream stream);

/// CHANNEL only; returns NotSupported for other backends.
/// At levels 2/3 all compute streams in this context's group are affected.
/// Independent copies may continue; use separate contexts for independent scheduling.
typedef struct {
    CUcontext context;
    int compute_group_scope;
    int independent_copies_may_run;
    int preserves_inflight_work;
    int supports_drop_commands;
} CudaChannelInfo;

XResult CudaQueueGetChannelInfo(HwQueueHandle hwq, CudaChannelInfo *info);

#ifdef __cplusplus
}
#endif
