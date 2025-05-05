#include "xsched/utils/common.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/preempt/xqueue/xqueue.h"

using namespace xsched::preempt;

std::mutex HwQueueManager::mtx_;
std::unordered_map<HwQueueHandle, std::shared_ptr<HwQueue>> HwQueueManager::hwqs_;

XResult HwQueueManager::Add(HwQueueHandle hwq_h, std::function<std::shared_ptr<HwQueue>()> create)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = hwqs_.find(hwq_h);
    if (it != hwqs_.end()) {
        XWARN("HwQueue with handle 0x%lx already exists", hwq_h);
        return kXSchedSuccess;
    }

    auto hwq = create();
    if (hwq == nullptr) {
        XWARN("Fail to create HwQueue with handle 0x%lx", hwq_h);
        return kXSchedErrorUnknown;
    }

    XASSERT(hwq->GetHandle() == hwq_h, "HwQueue handle mismatch");
    hwqs_[hwq_h] = hwq;
    return kXSchedSuccess;
}

XResult HwQueueManager::Del(HwQueueHandle hwq_h)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = hwqs_.find(hwq_h);
    if (it == hwqs_.end()) {
        XWARN("HwQueue with handle 0x%lx does not exist", hwq_h);
        return kXSchedErrorNotFound;
    }

    auto xq_shptr = it->second->GetXQueue();
    if (xq_shptr != nullptr) {
        XWARN("HwQueue (0x%lx) is still associated with XQueue (0x%lx), "
              "please first destroy the XQueue using XQueueDestroy()",
              hwq_h, xq_shptr->GetHandle());
        return kXSchedErrorNotAllowed;
    }

    hwqs_.erase(it);
    return kXSchedSuccess;
}

std::shared_ptr<HwQueue> HwQueueManager::Get(HwQueueHandle hwq_h)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = hwqs_.find(hwq_h);
    if (it == hwqs_.end()) return nullptr;
    return it->second;
}

std::shared_ptr<XQueue> HwQueueManager::GetXQueue(HwQueueHandle hwq_h)
{
    std::shared_ptr<HwQueue> hwq_shptr = Get(hwq_h);
    if (!hwq_shptr) return nullptr;
    return hwq_shptr->GetXQueue();
}

EXPORT_C_FUNC XResult HwQueueDestroy(HwQueueHandle hwq)
{
    return HwQueueManager::Del(hwq);
}

EXPORT_C_FUNC XResult HwQueueLaunch(HwQueueHandle hwq, HwCommandHandle hw_cmd)
{
    std::shared_ptr<HwQueue> hwq_shptr = HwQueueManager::Get(hwq);
    if (hwq_shptr == nullptr) {
        XWARN("HwQueue with handle 0x%lx does not exist", hwq);
        return kXSchedErrorNotFound;
    }
    // Get and DELETE the HwCommand from the HwCommandManager.
    std::shared_ptr<HwCommand> hw_cmd_shptr = HwCommandManager::Del(hw_cmd);
    if (hw_cmd_shptr == nullptr) {
        XWARN("HwCommand with handle 0x%lx does not exist or not registered", hw_cmd);
        return kXSchedErrorNotFound;
    }

    auto callback = std::dynamic_pointer_cast<HwCallbackCommand>(hw_cmd_shptr);
    if (callback != nullptr) return callback->Launch(hwq);

    hwq_shptr->Launch(hw_cmd_shptr);
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult HwQueueSynchronize(HwQueueHandle hwq)
{
    std::shared_ptr<HwQueue> hwq_shptr = HwQueueManager::Get(hwq);
    if (hwq_shptr == nullptr) {
        XWARN("HwQueue with handle 0x%lx does not exist", hwq);
        return kXSchedErrorNotFound;
    }
    hwq_shptr->Synchronize();
    return kXSchedSuccess;
}
