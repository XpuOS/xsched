#include "xsched/utils/xassert.h"
#include "xsched/preempt/sched/agent.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/preempt/xqueue/async_xqueue.h"

using namespace xsched::sched;
using namespace xsched::preempt;

AsyncXQueue::AsyncXQueue(std::shared_ptr<HwQueue> hwq, XPreemptLevel level,
                         int64_t threshold, int64_t batch_size)
    : XQueue(kXQueueImplTypeAsync,
             kQueueFeatureAsyncSubmit      |
             (hwq->SupportDynamicLevel()   ? kQueueFeatureDynamicLevel : 0)  |
             kQueueFeatureDynamicThreshold | kQueueFeatureDynamicBatchSize   |
             kQueueFeatureSyncSuspend      | kQueueFeatureResumeDropCommands ,
             hwq)
    , level_(level)
    , cmd_buf_(std::make_shared<CommandBuffer>(kHandle))
    , launch_worker_(std::make_shared<LaunchWorker>(hwq, cmd_buf_, level, threshold, batch_size))
{
    XASSERT(level_ > kPreemptLevelUnknown && level_ < kPreemptLevelMax,
            "invalid preempt level: %d", level_);

    auto wait_cmd = cmd_buf_->EnqueueXQueueWaitAllCommand();
    wait_cmd->Wait();
    kHwQueue->OnPreemptLevelChange(level_);
    SchedAgent::SendEvent(std::make_shared<XQueueCreateEvent>(kHandle, kDevice, level,
                                                              threshold, batch_size));
}

AsyncXQueue::~AsyncXQueue()
{
    // If the XQueue is terminated, it should not be preempted.
    terminated_.store(true);
    this->Resume(true);

    auto destroy_command = std::make_shared<XQueueDestroyCommand>();
    cmd_buf_->Enqueue(destroy_command);
    destroy_command->Wait();
    SchedAgent::SendEvent(std::make_shared<XQueueDestroyEvent>(kHandle));
}

void AsyncXQueue::Submit(std::shared_ptr<HwCommand> hw_cmd)
{
    hw_cmd->OnSubmit(shared_from_this());
    hw_cmd->SetIdx(next_hw_cmd_idx_.fetch_add(1));
    kHwQueue->OnHwCommandSubmit(hw_cmd);
    cmd_buf_->Enqueue(hw_cmd);
    if (hw_cmd->GetProps(kCommandPropertyBlockingSubmit)) hw_cmd->WaitUntil(kCommandStateInFlight);
}

std::shared_ptr<XQueueWaitAllCommand> AsyncXQueue::SubmitWaitAll()
{
    return cmd_buf_->EnqueueXQueueWaitAllCommand();
}

void AsyncXQueue::WaitAll()
{
    auto sync_command = cmd_buf_->EnqueueXQueueWaitAllCommand();
    sync_command->Wait();
}

void AsyncXQueue::Wait(std::shared_ptr<HwCommand> hw_cmd)
{
    if (!hw_cmd->Synchronizable()) {
        // If it is not synchronizable, it can only be synced
        // by waiting its state turns to kCommandStateCompleted.
        hw_cmd->WaitUntil(kCommandStateCompleted);
        return;
    }

    // Check if it is already completed.
    if (hw_cmd->GetState() >= kCommandStateCompleted) return;

    // If this HwCommand is synchronizable, then use launch worker to sync it.
    hw_cmd->WaitUntil(kCommandStateInFlight);
    launch_worker_->SyncCmd(hw_cmd);
}

XQueueState AsyncXQueue::Query()
{
    return cmd_buf_->GetXQueueState();
}

int64_t AsyncXQueue::GetHwCommandCount()
{
    return next_hw_cmd_idx_.load() - 1;
}

void AsyncXQueue::Suspend(int64_t flags)
{
    // If the XQueue is already terminated, it should not be suspended.
    if (terminated_.load()) return;

    // Will not suspend if it is already suspended.
    bool expected = false;
    if (!suspended_.compare_exchange_strong(expected, true)) return;

    launch_worker_->Pause();
    if (level_ >= kPreemptLevelDeactivate) kHwQueue->Deactivate();
    if (level_ >= kPreemptLevelInterrupt)  kHwQueue->Interrupt();
    if (flags & kQueueSuspendFlagSyncHwQueue) kHwQueue->Synchronize();
}

void AsyncXQueue::Resume(int64_t flags)
{
    // Will not resume if it is not suspended.
    bool expected = true;
    if (!suspended_.compare_exchange_strong(expected, false)) return;

    if (level_ == kPreemptLevelBlock) {
        // Should not clear the command log because when the XQueue is suspended without
        // synchronization, the HwCommand may be still running when resumes.
        launch_worker_->Resume();
        return;
    }

    // synchronization should be done in HwQueue::Reactivate.
    if (!(flags & kQueueResumeFlagDropCommands)) {
        // Regular restore, reactivate and then resume if not dropping commands.
        const CommandLog &log = launch_worker_->GetCommandLog();
        if (level_ >= kPreemptLevelInterrupt)  kHwQueue->Restore(log);
        if (level_ >= kPreemptLevelDeactivate) kHwQueue->Reactivate(log);
        launch_worker_->Resume();
        return;
    }

    // Restore and reactivate the hal queue with an empty command log.
    CommandLog empty_log = {};
    if (level_ >= kPreemptLevelInterrupt)  kHwQueue->Restore(empty_log);
    if (level_ >= kPreemptLevelDeactivate) kHwQueue->Reactivate(empty_log);

    // Drop all HwCommands in the command buffer.
    cmd_buf_->DropAll();

    // drop_idx is the last HwCommand idx that needs to be dropped.
    // Here drop_idx is the idx of the last HwCommand submitted.
    int64_t drop_idx = next_hw_cmd_idx_.load() - 1;
    launch_worker_->ResumeAndDrop(drop_idx);
}

void AsyncXQueue::SetPreemptLevel(XPreemptLevel level)
{
    XASSERT(level > kPreemptLevelUnknown && level < kPreemptLevelMax,
            "invalid preempt level: %d", level);
    // Dynamic level support of this AsyncXQueue depends on kHwQueue->SupportDynamicLevel().
    if (!this->GetFeatures(kQueueFeatureDynamicLevel)) {
        XASSERT(level == level_, "AsyncXQueue does not support dynamic level");
        return;
    }
    if (level > kHwQueue->GetMaxSupportedLevel()) {
        XWARN("preempt level %d is not supported by the HwQueue, "
              "max supported level is %d", level, kHwQueue->GetMaxSupportedLevel());
        return;
    }
    // Not thread-safe (with Submit & Wait).
    // TODO: change to thread-safe, because the level change may be triggered by scheduler or xcli.
    this->WaitAll();
    kHwQueue->OnPreemptLevelChange(level);
    launch_worker_->SetPreemptLevel(level);
    level_ = level;
    SchedAgent::SendEvent(std::make_shared<XQueueConfigUpdateEvent>(kHandle, kDevice, level_,
        launch_worker_->GetThreshold(), launch_worker_->GetBatchSize()));
}

void AsyncXQueue::SetLaunchConfig(int64_t threshold, int64_t batch_size)
{
    if (threshold <= 0 && batch_size <= 0) return;

    // Check command threshold.
    if (threshold <= 0) {
        threshold = launch_worker_->GetThreshold();
    } else {
        XASSERT(this->GetFeatures(kQueueFeatureDynamicThreshold),
                "AsyncXQueue does not support dynamic threshold");
    }

    // Check command batch size.
    if (batch_size <= 0) {
        batch_size = launch_worker_->GetBatchSize();
    } else {
        XASSERT(this->GetFeatures(kQueueFeatureDynamicBatchSize),
                "AsyncXQueue does not support dynamic batch size");
    }

    // Check threshold and batch size are valid.
    XASSERT(threshold >= batch_size,
            "command threshold (%ld) must not be smaller than command batch size (%ld)",
            threshold, batch_size);

    // Not thread-safe.
    // TODO: change to thread-safe
    this->WaitAll();
    launch_worker_->SetLaunchConfig(threshold, batch_size);
    SchedAgent::SendEvent(std::make_shared<XQueueConfigUpdateEvent>(kHandle, kDevice, level_,
        launch_worker_->GetThreshold(), launch_worker_->GetBatchSize()));
}
