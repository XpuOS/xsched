#pragma once

#include "xsched/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Create an XQueue.
/// @param xq    [out] Created handle to the XQueue.
/// @param hwq   [in]  Handle to the HwQueue which has been created by
/// the Hardware Abstraction Layer (HAL) of each platform, e.g., CudaQueueCreate().
/// @param level [in]  Preemption level, see XPreemptLevel.
/// @param flags [in]  XQueue creation flags, see XQueueCreateFlag.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult XQueueCreate(XQueueHandle *xq, HwQueueHandle hwq, int64_t level, int64_t flags);

/// @brief Destroy an XQueue.
/// @param xq [in] Handle to the XQueue.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult XQueueDestroy(XQueueHandle xq);

/// @brief Set the preemption level of an XQueue.
/// @param xq    [in] Handle to the XQueue.
/// @param level [in] Preemption level, see XPreemptLevel.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult XQueueSetPreemptLevel(XQueueHandle xq, int64_t level);

/// @brief Set the launch threshold and command batch size of an XQueue.
/// @param xq         [in] Handle to the XQueue.
/// @param threshold  [in] The number of commands that can be in flight.
/// @param batch_size [in] The number of commands to launch and synchronize at a time.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult XQueueSetLaunchConfig(XQueueHandle xq, int64_t threshold, int64_t batch_size);

/// @brief Submit an HwCommand to an XQueue.
/// @param xq     [in] Handle to the XQueue.
/// @param hw_cmd [in] Handle to the HwCommand which has been created by
/// the HAL of each platform, or HwCommandCreateCallback().
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult XQueueSubmit(XQueueHandle xq, HwCommandHandle hw_cmd);

/// @brief Wait for an HwCommand to complete on an XQueue.
/// @param xq     [in] Handle to the XQueue.
/// @param hw_cmd [in] Handle to the HwCommand which has been submitted to the XQueue.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult XQueueWait(XQueueHandle xq, HwCommandHandle hw_cmd);

/// @brief Wait for all HwCommands to complete on an XQueue.
/// @param xq [in] Handle to the XQueue.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult XQueueWaitAll(XQueueHandle xq);

/// @brief Query the state of an XQueue.
/// @param xq    [in]  Handle to the XQueue.
/// @param state [out] State of the XQueue, see XQueueState.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult XQueueQuery(XQueueHandle xq, XQueueState *state);

/// @brief Suspend an XQueue to stop its execution. Re-entrant.
/// @param xq    [in] Handle to the XQueue.
/// @param flags [in] Suspend flags, see XQueueSuspendFlag.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult XQueueSuspend(XQueueHandle xq, int64_t flags);

/// @brief Resume an XQueue to resume its execution. Re-entrant.
/// @param xq    [in] Handle to the XQueue.
/// @param flags [in] Resume flags, see XQueueResumeFlag.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult XQueueResume(XQueueHandle xq, int64_t flags);

/// @brief Get the number of HwCommands that have been submitted to an XQueue.
/// @param xq    [in]  Handle to the XQueue.
/// @param count [out] The number of HwCommands that have been submitted to the XQueue.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult XQueueProfileHwCommandCount(XQueueHandle xq, int64_t *count);

/// @brief Destroy an HwQueue.
/// @param hwq [in] Handle to the HwQueue which has been created by
/// the Hardware Abstraction Layer (HAL) of each platform, e.g., CudaQueueCreate().
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult HwQueueDestroy(HwQueueHandle hwq);

/// @brief Launch an HwCommand on an HwQueue.
/// @param hwq     [in] Handle to the HwQueue.
/// @param hw_cmd  [in] Handle to the HwCommand which has been created by the HAL of each platform.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult HwQueueLaunch(HwQueueHandle hwq, HwCommandHandle hw_cmd);

/// @brief Synchronize an HwQueue.
/// @param hwq [in] Handle to the HwQueue.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult HwQueueSynchronize(HwQueueHandle hwq);

/// @brief Create an HwCommand the executes a launch callback function.
/// @param hw_cmd [out] Handle to the HwCommand.
/// @param launch [in]  Launch callback function.
/// @param data   [in]  User data for the launch callback function.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult HwCommandCreateCallback(HwCommandHandle *hw_cmd, LaunchCallback launch, void *data);

/// @brief Destroy an HwCommand.
/// @param hw_cmd [in] Handle to the HwCommand.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult HwCommandDestroy(HwCommandHandle hw_cmd);

#ifdef __cplusplus
}
#endif
