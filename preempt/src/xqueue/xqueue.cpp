#include "xsched/hint.h"
#include "xsched/xqueue.h"
#include "xsched/preempt/hal/hw_command.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/preempt/xqueue/async_xqueue.h"

using namespace xsched::preempt;

std::mutex XQueueManager::mtx_;
std::unordered_map<XQueueHandle, std::shared_ptr<XQueue>> XQueueManager::xqs_;

XResult XQueueManager::Add(XQueueHandle *xq_hp, HwQueueHandle hwq_h, int64_t level, int64_t flags)
{
    if (xq_hp == nullptr) return kXSchedErrorInvalidValue;
    if (level <= kPreemptLevelUnknown || level >= kPreemptLevelMax) return kXSchedErrorInvalidValue;

    // TODO: implement BlockingXQueue
    if (flags & kQueueCreateFlagBlockingSubmit) return kXSchedErrorNotSupported;

    std::lock_guard<std::mutex> lock(mtx_);
    std::shared_ptr<HwQueue> hwq_shptr = HwQueueManager::Get(hwq_h);
    if (hwq_shptr == nullptr) {
        XWARN("HwQueue with handle 0x%lx does not exist or not registered", hwq_h);
        return kXSchedErrorNotFound;
    }

    if (hwq_shptr->GetXQueue() != nullptr) {
        XQueueHandle xq_h = hwq_shptr->GetXQueue()->GetHandle();
        auto it = xqs_.find(xq_h);
        XASSERT(it != xqs_.end(), "XQueue with handle 0x%lx does not exist", xq_h);
        XASSERT(it->second == hwq_shptr->GetXQueue(), "XQueue and handle 0x%lx mismatch", xq_h);
        XWARN("HwQueue (0x%lx) already has an XQueue (0x%lx)", hwq_h, xq_h);
        if (xq_hp != nullptr) *xq_hp = xq_h;
        return kXSchedSuccess;
    }

    auto xq_shptr = std::make_shared<AsyncXQueue>(hwq_shptr, (XPreemptLevel)level);
    if (xq_shptr == nullptr) {
        XWARN("Fail to create XQueue");
        return kXSchedErrorUnknown;
    }

    hwq_shptr->SetXQueue(xq_shptr); // Circular reference.
    XQueueHandle xq_h = xq_shptr->GetHandle();
    xqs_[xq_h] = xq_shptr;
    if (xq_hp != nullptr) *xq_hp = xq_h;
    return kXSchedSuccess;
}

XResult XQueueManager::Del(XQueueHandle xq_h)
{
    std::unique_lock<std::mutex> lock(mtx_);
    auto it = xqs_.find(xq_h);
    if (it == xqs_.end()) {
        XWARN("XQueue with handle 0x%lx does not exist", xq_h);
        return kXSchedErrorNotFound;
    }

    auto xq_shptr = it->second;
    auto hwq_shptr = xq_shptr->GetHwQueue();
    if (hwq_shptr != nullptr) hwq_shptr->SetXQueue(nullptr); // Break circular reference.
    xqs_.erase(it);
    lock.unlock(); // avoid xq_shptr->WaitAll(); with the lock held.

    // Wait for all commands to finish to prevent the launch worker from launching
    // commands on the HwQueue after the XQueue is deleted from the XQueueManager.
    xq_shptr->Resume(kQueueResumeFlagNone);
    xq_shptr->WaitAll();
    return kXSchedSuccess;
}

XResult XQueueManager::Exists(XQueueHandle xq_h)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = xqs_.find(xq_h);
    if (it == xqs_.end()) return kXSchedErrorNotFound;
    return kXSchedSuccess;
}

std::shared_ptr<XQueue> XQueueManager::Get(XQueueHandle xq_h)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = xqs_.find(xq_h);
    if (it == xqs_.end()) return nullptr;
    return it->second;
}

XResult XQueueManager::ForEachWaitAll()
{
    std::list<std::shared_ptr<XQueueWaitAllCommand>> wait_cmds;
    std::unique_lock<std::mutex> lock(mtx_);
    for (auto it : xqs_) {
        auto wait_cmd = it.second->SubmitWaitAll();
        if (wait_cmd == nullptr) return kXSchedErrorUnknown;
        wait_cmds.push_back(wait_cmd);
    }
    lock.unlock();
    for (auto &cmd : wait_cmds) cmd->Wait();
    return kXSchedSuccess;
}

XResult XQueueManager::ForEach(std::function<XResult(std::shared_ptr<XQueue>)> func)
{
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto it : xqs_) {
        XResult res = func(it.second);
        if (res != kXSchedSuccess) return res;
    }
    return kXSchedSuccess;
}

XResult XQueueManager::AutoCreate(std::function<XResult(HwQueueHandle *)> create_hwq)
{
    static char *env = std::getenv(XSCHED_AUTO_XQUEUE_ENV_NAME);
    if (env == nullptr || strlen(env) == 0 || strcmp(env, "0") == 0 ||
        strcasecmp(env, "off") == 0) {
        XDEBG("XQueue auto-create is disabled");
        return kXSchedSuccess;
    }
    HwCommandHandle hwq = 0;
    XResult res = create_hwq(&hwq);
    if (res != kXSchedSuccess) {
        XWARN("fail to auto-create HwQueue, err: %d", res);
        return res;
    }
    XDEBG("auto-created HwQueue 0x%lx", hwq);
    XQueueHandle xq = 0;
    XPreemptLevel level = XSCHED_DEFAULT_PREEMPT_LEVEL;

    // set level by env
    int64_t env_int64 = 0;
    bool env_set_level = GetEnvInt64(XSCHED_AUTO_XQUEUE_LEVEL_ENV_NAME, env_int64);
    if (env_set_level) level = (XPreemptLevel)env_int64;

    res = XQueueCreate(&xq, hwq, level, kQueueCreateFlagNone);
    if (res != kXSchedSuccess) {
        XWARN("fail to auto-create XQueue (level-%d) for HwQueue 0x%lx, err: %d",
              level, hwq, res);
        return res;
    }
    XDEBG("auto-created XQueue 0x%lx (level-%d) for HwQueue 0x%lx", xq, level, hwq);

    // set launch config by env
    int64_t env_threshold = -1;
    int64_t env_batch_size = -1;
    bool env_set_threshold = GetEnvInt64(XSCHED_AUTO_XQUEUE_THRESHOLD_ENV_NAME, env_threshold);
    bool env_set_batch_size = GetEnvInt64(XSCHED_AUTO_XQUEUE_BATCH_SIZE_ENV_NAME, env_batch_size);
    if (env_set_threshold || env_set_batch_size) {
        res = XQueueSetLaunchConfig(xq, env_threshold, env_batch_size);
        if (res != kXSchedSuccess) {
            XWARN("fail to auto-set launch config [%ld, %ld] for XQueue 0x%lx, err: %d",
                  env_threshold, env_batch_size, xq, res);
        }
        XDEBG("auto-set launch config [%ld, %ld] for XQueue 0x%lx",
              env_threshold, env_batch_size, xq);
    }

    // set priority by env
    int64_t env_priority = PRIORITY_DEFAULT;
    if (GetEnvInt64(XSCHED_AUTO_XQUEUE_PRIORITY_ENV_NAME, env_priority)) {
        res = XHintPriority(xq, env_priority);
        if (res != kXSchedSuccess) {
            XWARN("fail to auto-set priority %ld for XQueue 0x%lx, err: %d",
                  env_priority, xq, res);
        }
        XDEBG("auto-set priority %ld for XQueue 0x%lx", env_priority, xq);
    }

    // set utilization by env
    int64_t env_utilization;
    if (GetEnvInt64(XSCHED_AUTO_XQUEUE_UTILIZATION_ENV_NAME, env_utilization)) {
        res = XHintUtilization(xq, env_utilization);
        if (res != kXSchedSuccess) {
            XWARN("fail to auto-set utilization %ld for XQueue 0x%lx, err: %d",
                  env_utilization, xq, res);
        }
        XDEBG("auto-set utilization %ld %% for XQueue 0x%lx", env_utilization, xq);
    }

    // set timeslice by env
    int64_t env_timeslice;
    if (GetEnvInt64(XSCHED_AUTO_XQUEUE_TIMESLICE_ENV_NAME, env_timeslice)) {
        res = XHintTimeslice(env_timeslice);
        if (res != kXSchedSuccess) {
            XWARN("fail to auto-set timeslice %ld for XQueue 0x%lx, err: %d",
                  env_timeslice, xq, res);
        }
        XDEBG("auto-set timeslice %ld us", env_timeslice);
    }

    // set laxity by env
    int64_t env_laxity;
    if (GetEnvInt64(XSCHED_AUTO_XQUEUE_LAXITY_ENV_NAME, env_laxity)) {
        res = XHintLaxity(xq, env_laxity, PRIORITY_DEFAULT, env_priority);
        if (res != kXSchedSuccess) {
            XWARN("fail to auto-set laxity %ld for XQueue 0x%lx, err: %d",
                  env_laxity, xq, res);
        }
        XDEBG("auto-set laxity %ld, laxity-prio %d, crit-prio %ld",
              env_laxity, PRIORITY_DEFAULT, env_priority);
    }
    return res;
}

XResult XQueueManager::AutoDestroy(HwQueueHandle hwq_h)
{
    static char *env = std::getenv(XSCHED_AUTO_XQUEUE_ENV_NAME);
    if (env == nullptr || strlen(env) == 0 || strcmp(env, "0") == 0 ||
        strcasecmp(env, "off") == 0) {
        XDEBG("XQueue auto-destroy is disabled");
        return kXSchedSuccess;
    }
    auto xq_shptr = HwQueueManager::GetXQueue(hwq_h);
    if (xq_shptr == nullptr) {
        XWARN("XQueue for HwQueue 0x%lx does not exist", hwq_h);
        return kXSchedErrorNotFound;
    }
    XResult res = XQueueDestroy(xq_shptr->GetHandle());
    if (res != kXSchedSuccess) {
        XWARN("fail to auto-destroy XQueue 0x%lx for HwQueue 0x%lx, err: %d",
              xq_shptr->GetHandle(), hwq_h, res);
        return res;
    }
    XDEBG("auto-destroyed XQueue 0x%lx for HwQueue 0x%lx", xq_shptr->GetHandle(), hwq_h);
    res = HwQueueDestroy(hwq_h);
    if (res != kXSchedSuccess) {
        XWARN("fail to auto-destroy HwQueue 0x%lx, err: %d", hwq_h, res);
        return res;
    }
    XDEBG("auto-destroyed HwQueue 0x%lx", hwq_h);
    return res;
}

namespace xsched::preempt
{

static std::mutex xq_create_mtx_;

}

EXPORT_C_FUNC XResult XQueueCreate(XQueueHandle *xq, HwQueueHandle hwq, int64_t level, int64_t flags)
{
    return XQueueManager::Add(xq, hwq, level, flags);
}

EXPORT_C_FUNC XResult XQueueDestroy(XQueueHandle xq)
{
    return XQueueManager::Del(xq);
}

EXPORT_C_FUNC XResult XQueueSetPreemptLevel(XQueueHandle xq, int64_t level)
{
    std::shared_ptr<XQueue> xq_shptr = XQueueManager::Get(xq);
    if (xq_shptr == nullptr) {
        XWARN("XQueue with handle 0x%lx does not exist", xq);
        return kXSchedErrorNotFound;
    }
    if (!xq_shptr->GetFeatures(kQueueFeatureDynamicLevel)) {
        XWARN("XQueue with handle 0x%lx does not support dynamic level", xq);
        return kXSchedErrorNotSupported;
    }
    xq_shptr->SetPreemptLevel((XPreemptLevel)level);
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult XQueueSetLaunchConfig(XQueueHandle xq, int64_t threshold, int64_t batch_size)
{
    std::shared_ptr<XQueue> xq_shptr = XQueueManager::Get(xq);
    if (xq_shptr == nullptr) {
        XWARN("XQueue with handle 0x%lx does not exist", xq);
        return kXSchedErrorNotFound;
    }

    if (threshold <= 0 && batch_size <= 0) return kXSchedSuccess;
    if (threshold > 0 && !xq_shptr->GetFeatures(kQueueFeatureDynamicThreshold)) {
        XWARN("XQueue with handle 0x%lx does not support dynamic command threshold", xq);
        return kXSchedErrorNotSupported;
    }
    if (batch_size > 0 && !xq_shptr->GetFeatures(kQueueFeatureDynamicBatchSize)) {
        XWARN("XQueue with handle 0x%lx does not support dynamic command batch size", xq);
        return kXSchedErrorNotSupported;
    }

    xq_shptr->SetLaunchConfig(threshold, batch_size);
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult XQueueSubmit(XQueueHandle xq, HwCommandHandle hw_cmd)
{
    std::shared_ptr<XQueue> xq_shptr = XQueueManager::Get(xq);
    if (xq_shptr == nullptr) {
        XWARN("XQueue with handle 0x%lx does not exist", xq);
        return kXSchedErrorNotFound;
    }
    // Get and DELETE the HwCommand from the HwCommandManager.
    std::shared_ptr<HwCommand> hw_cmd_shptr = HwCommandManager::Del(hw_cmd);
    if (hw_cmd_shptr == nullptr) {
        XWARN("HwCommand with handle 0x%lx does not exist or not registered", hw_cmd);
        return kXSchedErrorNotFound;
    }
    xq_shptr->Submit(hw_cmd_shptr);
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult XQueueWait(XQueueHandle xq, HwCommandHandle hw_cmd)
{
    std::shared_ptr<XQueue> xq_shptr = XQueueManager::Get(xq);
    if (xq_shptr == nullptr) {
        XWARN("XQueue with handle 0x%lx does not exist", xq);
        return kXSchedErrorNotFound;
    }
    std::shared_ptr<HwCommand> hw_cmd_shptr = HwCommandManager::Get(hw_cmd);
    if (hw_cmd_shptr == nullptr) {
        XWARN("HwCommand with handle 0x%lx does not exist or not registered", hw_cmd);
        return kXSchedErrorNotFound;
    }
    xq_shptr->Wait(hw_cmd_shptr);
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult XQueueWaitAll(XQueueHandle xq)
{
    std::shared_ptr<XQueue> xq_shptr = XQueueManager::Get(xq);
    if (xq_shptr == nullptr) {
        XWARN("XQueue with handle 0x%lx does not exist", xq);
        return kXSchedErrorNotFound;
    }
    xq_shptr->WaitAll();
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult XQueueQuery(XQueueHandle xq, XQueueState *state)
{
    if (state == nullptr) return kXSchedErrorInvalidValue;
    std::shared_ptr<XQueue> xq_shptr = XQueueManager::Get(xq);
    if (xq_shptr == nullptr) {
        XWARN("XQueue with handle 0x%lx does not exist", xq);
        return kXSchedErrorNotFound;
    }
    *state = xq_shptr->Query();
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult XQueueSuspend(XQueueHandle xq, int64_t flags)
{
    std::shared_ptr<XQueue> xq_shptr = XQueueManager::Get(xq);
    if (xq_shptr == nullptr) {
        XWARN("XQueue with handle 0x%lx does not exist", xq);
        return kXSchedErrorNotFound;
    }
    xq_shptr->Suspend(flags);
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult XQueueResume(XQueueHandle xq, int64_t flags)
{
    std::shared_ptr<XQueue> xq_shptr = XQueueManager::Get(xq);
    if (xq_shptr == nullptr) {
        XWARN("XQueue with handle 0x%lx does not exist", xq);
        return kXSchedErrorNotFound;
    }
    xq_shptr->Resume(flags);
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult XQueueProfileHwCommandCount(XQueueHandle xq, int64_t *count)
{
    if (count == nullptr) return kXSchedErrorInvalidValue;
    std::shared_ptr<XQueue> xq_shptr = XQueueManager::Get(xq);
    if (xq_shptr == nullptr) {
        XWARN("XQueue with handle 0x%lx does not exist", xq);
        return kXSchedErrorNotFound;
    }
    *count = xq_shptr->GetHwCommandCount();
    return kXSchedSuccess;
}
