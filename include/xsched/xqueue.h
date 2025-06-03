#pragma once

#include "xsched/types.h"

#ifdef __cplusplus
extern "C" {
#endif

XResult XQueueCreate(XQueueHandle *xq, HwQueueHandle hwq, int64_t level, int64_t flags);
XResult XQueueDestroy(XQueueHandle xq);

XResult XQueueSetPreemptLevel(XQueueHandle xq, int64_t level);
XResult XQueueSetLaunchConfig(XQueueHandle xq, int64_t threshold, int64_t batch_size);

XResult XQueueSubmit(XQueueHandle xq, HwCommandHandle hw_cmd);
XResult XQueueWait(XQueueHandle xq, HwCommandHandle hw_cmd);
XResult XQueueWaitAll(XQueueHandle xq);
XResult XQueueQuery(XQueueHandle xq, XQueueState *state);

XResult XQueueSuspend(XQueueHandle xq, int64_t flags);
XResult XQueueResume(XQueueHandle xq, int64_t flags);

XResult XQueueProfileHwCommandCount(XQueueHandle xq, int64_t *count);

// HwQueues are created by the XPU hardware abstraction layer (HAL).
XResult HwQueueDestroy(HwQueueHandle hwq);

XResult HwQueueLaunch(HwQueueHandle hwq, HwCommandHandle hw_cmd);
XResult HwQueueSynchronize(HwQueueHandle hwq);

XResult HwCommandCreateCallback(HwCommandHandle *hw_cmd, LaunchCallback launch, void *data);
XResult HwCommandDestroy(HwCommandHandle hw_cmd);

#ifdef __cplusplus
}
#endif
