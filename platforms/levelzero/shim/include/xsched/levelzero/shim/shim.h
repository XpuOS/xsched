#pragma once

#include "xsched/levelzero/hal/driver.h"

namespace xsched::levelzero
{

ze_result_t XCommandQueueCreate(ze_context_handle_t hContext, ze_device_handle_t hDevice, const ze_command_queue_desc_t *desc, ze_command_queue_handle_t *phCommandQueue);
ze_result_t XCommandQueueDestroy(ze_command_queue_handle_t hCommandQueue);
ze_result_t XCommandQueueExecuteCommandLists(ze_command_queue_handle_t hCommandQueue, uint32_t numCommandLists, ze_command_list_handle_t *phCommandLists, ze_fence_handle_t hFence);
ze_result_t XCommandQueueSynchronize(ze_command_queue_handle_t hCommandQueue, uint64_t timeout);

ze_result_t XCommandListCreate(ze_context_handle_t hContext, ze_device_handle_t hDevice, const ze_command_list_desc_t *desc, ze_command_list_handle_t *phCommandList);
ze_result_t XCommandListDestroy(ze_command_list_handle_t hCommandList);
ze_result_t XCommandListClose(ze_command_list_handle_t hCommandList);
ze_result_t XCommandListReset(ze_command_list_handle_t hCommandList);
ze_result_t XCommandListAppendMemoryCopy(ze_command_list_handle_t hCommandList, void *dstptr, const void *srcptr, size_t size, ze_event_handle_t hSignalEvent, uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);
ze_result_t XCommandListAppendLaunchKernel(ze_command_list_handle_t hCommandList, ze_kernel_handle_t hKernel, const ze_group_count_t *pLaunchFuncArgs, ze_event_handle_t hSignalEvent, uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);

ze_result_t XFenceDestroy(ze_fence_handle_t hFence);
ze_result_t XFenceHostSynchronize(ze_fence_handle_t hFence, uint64_t timeout);
ze_result_t XFenceQueryStatus(ze_fence_handle_t hFence);
ze_result_t XFenceReset(ze_fence_handle_t hFence);

ze_result_t XEventDestroy(ze_event_handle_t hEvent);
ze_result_t XEventHostSignal(ze_event_handle_t hEvent);
ze_result_t XEventHostSynchronize(ze_event_handle_t hEvent, uint64_t timeout);
ze_result_t XEventQueryStatus(ze_event_handle_t hEvent);
ze_result_t XEventHostReset(ze_event_handle_t hEvent);

} // namespace xsched::levelzero
