#pragma once

#include "xsched/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _ze_device_handle_t *ze_device_handle_t;
typedef struct _ze_command_queue_handle_t *ze_command_queue_handle_t;

// create a HwQueue
XResult ZeQueueCreate(HwQueueHandle *hwq, ze_device_handle_t dev, ze_command_queue_handle_t cmdq);

// create HwCommands

#ifdef __cplusplus
}
#endif
