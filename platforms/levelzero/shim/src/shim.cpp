#include "xsched/utils/map.h"
#include "xsched/levelzero/hal.h"
#include "xsched/levelzero/hal/types.h"
#include "xsched/levelzero/hal/driver.h"
#include "xsched/levelzero/hal/ze_command.h"
#include "xsched/levelzero/shim/shim.h"
#include "xsched/levelzero/shim/cmd_list.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/preempt/xqueue/xqueue.h"

using namespace xsched::utils;
using namespace xsched::preempt;

namespace xsched::levelzero
{

static ObjectMap<ze_event_handle_t, ze_command_list_handle_t> g_event_to_cmdlist;
static ObjectMap<ze_fence_handle_t, std::shared_ptr<ZeListExecuteCommand>> g_fence_to_cmd;
static ObjectMap<ze_command_list_handle_t, std::shared_ptr<ZeListExecuteCommand>> g_cmdlist_to_cmd;
static ObjectMap<ze_event_handle_t, std::shared_ptr<ZeKernelCommand>> g_event_to_cmd;

ze_result_t XCommandQueueCreate(ze_context_handle_t hContext, ze_device_handle_t hDevice, const ze_command_queue_desc_t *desc, ze_command_queue_handle_t *phCommandQueue)
{
    ze_result_t res = Driver::CommandQueueCreate(hContext, hDevice, desc, phCommandQueue);
    if (res != ZE_RESULT_SUCCESS) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return ZeQueueCreate(hwq, hDevice, *phCommandQueue); });
    XDEBG("XCommandQueueCreate(cmdq: %p, dev: %p)", *phCommandQueue, hDevice);
    return res;
}

ze_result_t XCommandQueueDestroy(ze_command_queue_handle_t hCommandQueue)
{
    XDEBG("XCommandQueueDestroy(cmdq: %p)", hCommandQueue);
    XQueueManager::AutoDestroy(GetHwQueueHandle(hCommandQueue));
    return Driver::CommandQueueDestroy(hCommandQueue);
}

ze_result_t XCommandQueueExecuteCommandLists(ze_command_queue_handle_t cmdq, uint32_t num, ze_command_list_handle_t *cmd_lists, ze_fence_handle_t fence)
{
    for (uint32_t i = 0; i < num; i++) {
        XDEBG("XCommandQueueExecuteCommandLists(cmdq: %p, cmd_list: %p, fence: %p)", cmdq, cmd_lists[i], fence);
    }

    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(cmdq));
    if (xq == nullptr) return Driver::CommandQueueExecuteCommandLists(cmdq, num, cmd_lists, fence);

    std::list<std::shared_ptr<ZeListExecuteCommand>> cmds;
    for (uint32_t i = 0; i < num; i++) {
        auto sliced = CommandListManager::Instance().Get(cmd_lists[i]);
        if (sliced == nullptr || sliced->cmd_lists.empty()) {
            // no sliced, submit the true command list
            auto cmd = std::make_shared<ZeListExecuteCommand>(cmdq, cmd_lists[i]);
            g_cmdlist_to_cmd.Add(cmd_lists[i], cmd);
            cmds.push_back(cmd);
            continue;
        }
        // submit each slice
        for (auto &cl : sliced->cmd_lists) {
            auto cmd = std::make_shared<ZeListExecuteCommand>(cmdq, cl);
            g_cmdlist_to_cmd.Add(cl, cmd);
            cmds.push_back(cmd);
        }
    }

    for (auto it = cmds.begin(); it != cmds.end(); it++) {
        if (fence != nullptr && std::next(it) == cmds.end()) {
            XASSERT((*it)->EnableSynchronization(), "fail to enable synchronization");
            g_fence_to_cmd.Add(fence, *it);
        }
        xq->Submit(*it);
    }

    return ZE_RESULT_SUCCESS;
}

ze_result_t XCommandQueueSynchronize(ze_command_queue_handle_t hCommandQueue, uint64_t timeout)
{
    XDEBG("XCommandQueueSynchronize(cmdq: %p, timeout: " FMT_64D ")", hCommandQueue, timeout);
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(hCommandQueue));
    if (xq == nullptr) return Driver::CommandQueueSynchronize(hCommandQueue, timeout);
    xq->WaitAll();
    return ZE_RESULT_SUCCESS;
}

ze_result_t XCommandListCreate(ze_context_handle_t hContext, ze_device_handle_t hDevice, const ze_command_list_desc_t *desc, ze_command_list_handle_t *phCommandList)
{
    auto res = CommandListManager::Instance().Create(hContext, hDevice, desc, phCommandList);
    if (res != ZE_RESULT_SUCCESS) return res;
    XDEBG("XCommandListCreate(cmd_list: %p)", *phCommandList);
    return res;
}

ze_result_t XCommandListCreateImmediate(ze_context_handle_t hContext, ze_device_handle_t hDevice, const ze_command_queue_desc_t *altdesc, ze_command_list_handle_t *phCommandList)
{
    auto res = ImmediateCommandListManager::Instance().Create(hContext, hDevice, altdesc, phCommandList);
    if (res != ZE_RESULT_SUCCESS) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return ZeListreate(hwq, hDevice, *phCommandList); });
    XDEBG("XCommandListCreateImmediate(cmd_list: %p)", *phCommandList);
    return res;
}

ze_result_t XCommandListDestroy(ze_command_list_handle_t hCommandList)
{
    XDEBG("XCommandListDestroy(cmd_list: %p)", hCommandList);
    if (ImmediateCommandListManager::Instance().Exists(hCommandList)) {
        XQueueManager::AutoDestroy(GetHwQueueHandle(hCommandList));
        return ImmediateCommandListManager::Instance().Destroy(hCommandList);
    }

    auto sliced = CommandListManager::Instance().Get(hCommandList);
    if (sliced == nullptr) {
        g_cmdlist_to_cmd.Del(hCommandList, nullptr);
        return Driver::CommandListDestroy(hCommandList);
    }
    for (auto &cl : sliced->cmd_lists) g_cmdlist_to_cmd.Del(cl, nullptr);
    return CommandListManager::Instance().Destroy(hCommandList);
}

ze_result_t XCommandListReset(ze_command_list_handle_t hCommandList)
{
    XDEBG("XCommandListReset(cmd_list: %p)", hCommandList);
    auto sliced = CommandListManager::Instance().Get(hCommandList);
    if (sliced == nullptr) {
        g_cmdlist_to_cmd.Del(hCommandList, nullptr);
        return Driver::CommandListReset(hCommandList);
    }
    for (auto &cl : sliced->cmd_lists) g_cmdlist_to_cmd.Del(cl, nullptr);
    return CommandListManager::Instance().Reset(hCommandList);
}

ze_result_t XCommandListClose(ze_command_list_handle_t hCommandList)
{
    XDEBG("XCommandListClose(cmd_list: %p)", hCommandList);
    if (ImmediateCommandListManager::Instance().Exists(hCommandList))
        return Driver::CommandListClose(hCommandList);
    else
        return CommandListManager::Instance().Close(hCommandList);
}

ze_result_t XCommandListHostSynchronize(ze_command_list_handle_t hCommandList, uint64_t timeout)
{
    XDEBG("XCommandListHostSynchronize(cmd_list: %p, timeout: " FMT_64D ")", hCommandList, timeout);
    if (ImmediateCommandListManager::Instance().Exists(hCommandList)) {
        auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(hCommandList));
        if (xq == nullptr) return Driver::CommandListHostSynchronize(hCommandList, timeout);
        xq->WaitAll();
        return ZE_RESULT_SUCCESS;
    }

    XERRO("XCommandListHostSynchronize is not supported for non-immediate command list (%p)", hCommandList);
    return ZE_RESULT_ERROR_UNKNOWN;
}

ze_result_t XKernelSetGroupSize(ze_kernel_handle_t hKernel, uint32_t groupSizeX, uint32_t groupSizeY, uint32_t groupSizeZ)
{
    return KernelArgsManager::Instance().AddGroupSize(hKernel, groupSizeX, groupSizeY, groupSizeZ);
}
ze_result_t XKernelSetArgumentValue(ze_kernel_handle_t hKernel, uint32_t argIndex, size_t argSize, const void *pArgValue)
{
    return KernelArgsManager::Instance().AddArgumentValue(hKernel, argIndex, argSize, pArgValue);
}
ze_result_t XKernelSetIndirectAccess(ze_kernel_handle_t hKernel, ze_kernel_indirect_access_flags_t flags)
{
    return KernelArgsManager::Instance().AddIndirectAccess(hKernel, flags);
}
ze_result_t XKernelSetCacheConfig(ze_kernel_handle_t hKernel, ze_cache_config_flags_t flags)
{
    return KernelArgsManager::Instance().AddCacheConfig(hKernel, flags);
}
ze_result_t XKernelSetGlobalOffsetExp(ze_kernel_handle_t hKernel, uint32_t offsetX, uint32_t offsetY, uint32_t offsetZ)
{
    return KernelArgsManager::Instance().AddGlobalOffsetExp(hKernel, offsetX, offsetY, offsetZ);
}

ze_result_t XCommandListAppendLaunchKernel(ze_command_list_handle_t hCommandList, ze_kernel_handle_t hKernel, const ze_group_count_t *pLaunchFuncArgs, ze_event_handle_t hSignalEvent, uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents)
{
    KernelArgsManager::Instance().Freeze(hKernel);

    if (ImmediateCommandListManager::Instance().Exists(hCommandList)) {
        auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(hCommandList));
        if (xq == nullptr) {
            ZE_ASSERT(KernelArgsManager::Instance().Set(hKernel));
            return Driver::CommandListAppendLaunchKernel(hCommandList, hKernel, pLaunchFuncArgs, hSignalEvent, numWaitEvents, phWaitEvents);
        }
        auto cmd = std::make_shared<ZeKernelCommand>(hCommandList, hKernel, pLaunchFuncArgs, hSignalEvent, numWaitEvents, phWaitEvents);
        g_event_to_cmd.Add(hSignalEvent, cmd);
        xq->Submit(cmd);
        XDEBG("XCommandListAppendLaunchKernel(cmd_list: %p, kernel: %p) immediate submitted", hCommandList, hKernel);
        return ZE_RESULT_SUCCESS;
    }

    ZE_ASSERT(KernelArgsManager::Instance().Set(hKernel));
    return CommandListManager::Instance().Append(hCommandList, [&](ze_command_list_handle_t cmd_list) {
        if (hSignalEvent != nullptr) g_event_to_cmdlist.Add(hSignalEvent, cmd_list);
        return Driver::CommandListAppendLaunchKernel(cmd_list, hKernel, pLaunchFuncArgs, hSignalEvent, numWaitEvents, phWaitEvents);
    });
}

ze_result_t XFenceDestroy(ze_fence_handle_t hFence)
{
    XDEBG("XFenceDestroy(fence: %p)", hFence);
    g_fence_to_cmd.Del(hFence, nullptr);
    return Driver::FenceDestroy(hFence);
}

ze_result_t XFenceReset(ze_fence_handle_t hFence)
{
    XDEBG("XFenceReset(fence: %p)", hFence);
    g_fence_to_cmd.Del(hFence, nullptr);
    return Driver::FenceReset(hFence);
}

ze_result_t XFenceHostSynchronize(ze_fence_handle_t hFence, uint64_t timeout)
{
    XDEBG("XFenceHostSynchronize(fence: %p, timeout: " FMT_64D ")", hFence, timeout);
    auto cmd = g_fence_to_cmd.Get(hFence, nullptr);
    if (cmd == nullptr) return Driver::FenceHostSynchronize(hFence, timeout);
    XDEBG("XFenceHostSynchronize(fence: %p, cmd: %p)", hFence, cmd.get());
    cmd->Wait();
    return ZE_RESULT_SUCCESS;
}

ze_result_t XFenceQueryStatus(ze_fence_handle_t hFence)
{
    XDEBG("XFenceQueryStatus(fence: %p)", hFence);
    auto cmd = g_fence_to_cmd.Get(hFence, nullptr);
    if (cmd == nullptr) return Driver::FenceQueryStatus(hFence);
    ze_fence_handle_t fence = cmd->GetFollowingFence();
    XASSERT(fence != nullptr, "fence is nullptr");
    return Driver::FenceQueryStatus(fence);
}

ze_result_t XEventDestroy(ze_event_handle_t hEvent)
{
    g_event_to_cmdlist.Del(hEvent, nullptr);
    g_event_to_cmd.Del(hEvent, nullptr);
    return Driver::EventDestroy(hEvent);
}

ze_result_t XEventHostReset(ze_event_handle_t hEvent)
{
    g_event_to_cmdlist.Del(hEvent, nullptr);
    g_event_to_cmd.Del(hEvent, nullptr);
    return Driver::EventHostReset(hEvent);
}

ze_result_t XEventHostSignal(ze_event_handle_t hEvent)
{
    ze_command_list_handle_t cmdlist = g_event_to_cmdlist.Get(hEvent, nullptr);
    if (cmdlist == nullptr) {
        auto cmd = g_event_to_cmd.Get(hEvent, nullptr);
        if (cmd == nullptr) return Driver::EventHostSignal(hEvent);
        return cmd->EventHostSignal();
    }

    auto cmd = g_cmdlist_to_cmd.Get(cmdlist, nullptr);
    if (cmd == nullptr) return Driver::EventHostSignal(hEvent);
    cmd->WaitUntil(kCommandStateInFlight);
    return Driver::EventHostSignal(hEvent);
}

ze_result_t XEventHostSynchronize(ze_event_handle_t hEvent, uint64_t timeout)
{
    ze_command_list_handle_t cmdlist = g_event_to_cmdlist.Get(hEvent, nullptr);
    std::shared_ptr<HwCommand> cmd;
    if (cmdlist == nullptr)
        cmd = g_event_to_cmd.Get(hEvent, nullptr);
    else
        cmd = g_cmdlist_to_cmd.Get(cmdlist, nullptr);
    if (cmd == nullptr) return Driver::EventHostSynchronize(hEvent, timeout);
    cmd->Wait();
    return ZE_RESULT_SUCCESS;
}

ze_result_t XEventQueryStatus(ze_event_handle_t hEvent)
{
    ze_command_list_handle_t cmdlist = g_event_to_cmdlist.Get(hEvent, nullptr);
    if (cmdlist == nullptr) {
        auto cmd = g_event_to_cmd.Get(hEvent, nullptr);
        if (cmd == nullptr) return Driver::EventQueryStatus(hEvent);
        return cmd->EventQueryStatus();
    }

    auto cmd = g_cmdlist_to_cmd.Get(cmdlist, nullptr);
    if (cmd == nullptr) return Driver::EventQueryStatus(hEvent);
    cmd->WaitUntil(kCommandStateInFlight);
    return Driver::EventQueryStatus(hEvent);
}

} // namespace xsched::levelzero
