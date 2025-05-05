#include <memory>

#include "xsched/xqueue.h"
#include "xsched/utils/xassert.h"
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
    if (!executing_.load()) return;

    switch (op->Type())
    {
    case kOperationSched:
        ExecuteSchedOperation(std::dynamic_pointer_cast<const sched::SchedOperation>(op));
        break;
    case kOperationConfig:
        ExecuteConfigOperation(std::dynamic_pointer_cast<const sched::ConfigOperation>(op));
        break;
    default:
        XASSERT(false, "unknown operation type: %d", op->Type());
        break;
    }
}

void SchedExecutor::ExecuteSchedOperation(std::shared_ptr<const sched::SchedOperation> op)
{
    XASSERT(op != nullptr, "sched operation type mismatch");
    size_t running_cnt = op->RunningCnt();
    size_t suspended_cnt = op->SuspendedCnt();
    const XQueueHandle *handles = op->Handles();

    for (size_t i = 0; i < running_cnt; ++i) {
        std::shared_ptr<XQueue> xq_shptr = XQueueManager::Get(handles[i]);
        // It is possible that the XQueue has been destroyed because the operation is asynchronous.
        if (xq_shptr != nullptr) xq_shptr->Resume(kQueueResumeFlagNone);
    }
    for (size_t i = 0; i < suspended_cnt; ++i) {
        std::shared_ptr<XQueue> xq_shptr = XQueueManager::Get(handles[running_cnt + i]);
        if (xq_shptr != nullptr) xq_shptr->Suspend(kQueueSuspendFlagNone);
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
            XWARN("XQueueSetPreemptLevel failed, xq: 0x%lx, level: %d, result: %d", handle, level, res);
        }
    }

    if (threshold > 0 || batch_size > 0) {
        XResult res = XQueueSetLaunchConfig(handle, threshold, batch_size);
        if (res != kXSchedSuccess) {
            XWARN("XQueueSetThreshold failed, xq: 0x%lx, threshold: %ld, batch size: %ld, result: %d",
                  handle, threshold, batch_size, res);
        }
    }
}
