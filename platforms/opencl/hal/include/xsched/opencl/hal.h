#pragma once

#include "xsched/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _cl_command_queue *  cl_command_queue;

XResult OclQueueCreate(HwQueueHandle *hwq, cl_command_queue cmdq);

#ifdef __cplusplus
}
#endif
