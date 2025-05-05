#pragma once

#include "xsched/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _cl_command_queue *  cl_command_queue;

XResult SetOverrideCommandQueue(int32_t queue_idx);
XResult UnsetOverrideCommandQueue();
XResult GetOverrideCommandQueue(int32_t queue_idx, cl_command_queue *cmdq);
XResult DeleteOverrideCommandQueue(int32_t queue_idx);

#ifdef __cplusplus
}
#endif
