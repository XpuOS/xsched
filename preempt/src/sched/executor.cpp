#include <memory>

#include "xsched/xqueue.h"
#include "xsched/utils/env.h"
#include "xsched/utils/xassert.h"
#include "xsched/protocol/def.h"
#include "xsched/preempt/sched/agent.h"
#include "xsched/preempt/sched/executor.h"
#include "xsched/preempt/xqueue/xqueue.h"

using namespace xsched::sched;
using namespace xsched::preempt;

std::atomic_bool SchedExecutor::executing_(false);

void SchedExecutor::Start()
{
    executing_.store(true);
}

void SchedExecutor::Stop()
{
    executing_.store(false);
}

void SchedExecutor::Execute(std::shared_ptr<const sched::Operation> op)
{
    OperationType type = op->Type();
    XASSERT(op->Pid() == GetProcessId(),
            "operation %ld sent to the wrong process, target: %d, receiver %d",
            op->Id(), op->Pid(), GetProcessId());
    if (!executing_.load()) type = kOperationNone;

    switch (type)
    {
    case kOperationNone:
        break;
    case kOperationSched:
        ExecuteSchedOperation(std::dynamic_pointer_cast<const sched::SchedOperation>(op));
        break;
    case kOperationConfig:
        ExecuteConfigOperation(std::dynamic_pointer_cast<const sched::ConfigOperation>(op));
        break;
    default:
        XERRO("unsupported operation type: %d", type);
        break;
    }

    // report operation complete event to scheduler
    SchedAgent::SendEvent(std::make_shared<OperationCompleteEvent>(op->Id()));
}

void SchedExecutor::ExecuteSchedOperation(std::shared_ptr<const sched::SchedOperation> op)
{
    XASSERT(op != nullptr, "sched operation type mismatch");
    size_t running_cnt = op->RunningCnt();
    size_t suspended_cnt = op->SuspendedCnt();
    const XQueueHandle *handles = op->Handles();
    static const int64_t suspend_flags = 
        (GetEnvOption(XSCHED_SCHEDULER_SUSPEND_SYNC_HWQ_ENV_NAME, false)
            ? kQueueSuspendFlagSyncHwQueue : kQueueSuspendFlagNone)
      | (GetEnvOption(XSCHED_SCHEDULER_SUSPEND_WAIT_ALL_ENV_NAME, false)
            ? kQueueSuspendFlagWaitAll     : kQueueSuspendFlagNone)
      | (GetEnvOption(XSCHED_SCHEDULER_SUSPEND_WAIT_IDLE_ENV_NAME, false)
            ? kQueueSuspendFlagWaitIdle    : kQueueSuspendFlagNone);

    for (size_t i = 0; i < running_cnt; ++i) {
        std::shared_ptr<XQueue> xq_shptr = XQueueManager::Get(handles[i]);
        // It is possible that the XQueue has been destroyed because the operation is asynchronous.
        if (xq_shptr != nullptr) xq_shptr->Resume(kQueueResumeFlagNone);
    }
    for (size_t i = 0; i < suspended_cnt; ++i) {
        std::shared_ptr<XQueue> xq_shptr = XQueueManager::Get(handles[running_cnt + i]);
        if (xq_shptr != nullptr) xq_shptr->Suspend(suspend_flags);
    }
}

void SchedExecutor::ExecuteConfigOperation(std::shared_ptr<const sched::ConfigOperation> op)
{
    XASSERT(op != nullptr, "config operation type mismatch");
    XQueueHandle handle = op->Handle();
    XPreemptLevel level = op->Level();
    int64_t threshold = op->Threshold();
    int64_t batch_size = op->BatchSize();

    if (level > kPreemptLevelUnknown) {
        XResult res = XQueueSetPreemptLevel(handle, level);
        if (res != kXSchedSuccess) {
            XWARN("XQueueSetPreemptLevel failed, xq: 0x" FMT_64X ", level: %d, result: %d",
                  handle, level, res);
        }
    }

    if (threshold > 0 || batch_size > 0) {
        XResult res = XQueueSetLaunchConfig(handle, threshold, batch_size);
        if (res != kXSchedSuccess) {
            XWARN("XQueueSetThreshold failed, xq: 0x" FMT_64X ", threshold: " FMT_64D
                  ", batch size: " FMT_64D ", result: %d", handle, threshold, batch_size, res);
        }
    }
}
