#include "xsched/utils/xassert.h"
#include "xsched/preempt/xqueue/launch_worker.h"

using namespace xsched::utils;
using namespace xsched::preempt;

LaunchWorker::LaunchWorker(std::shared_ptr<HwQueue> hwq, std::shared_ptr<CommandBuffer> buf,
                           XPreemptLevel level, int64_t threshold, int64_t batch_size)
    : level_(level), threshold_(threshold), batch_size_(batch_size)
    , kHwq(hwq), kCmdBuf(buf), mtx_(std::make_unique<MCSLock>())
{
    // In-flight commands threshold must not be smaller than command batch size.
    // Otherwise, there could be no commands to wait if the in-flight commands reach the threshold.
    XASSERT(threshold_ >= batch_size_, "threshold must not be smaller than command batch size");

    worker_thread_ = std::make_unique<std::thread>([this](){
        // Notify the HwQueue that an XQueue has created for it on the submitting thread.
        // It can call platform specific APIs like cuCtxSetCurrent() in OnXQueueCreate().
        this->kHwq->OnXQueueCreate();
        this->WorkerLoop();
    });
}

LaunchWorker::~LaunchWorker()
{
    // When destroying the XQueue holding this LaunchWorker, the XQueue will submit an
    // XQueueDestroyCommand. When the worker thread consumes the XQueueDestroyCommand, it will call
    // SyncAllWithLock() and wait for all HwCommands to complete, and then clear data and exit.
    // ~LaunchWorker() should join the worker thread.
    worker_thread_->join();
}

void LaunchWorker::Pause()
{
    mtx_->lock();
    state_ = kWorkerStatePaused;
    pause_count_ += 1;
    mtx_->unlock();
}

void LaunchWorker::Resume()
{
    mtx_->lock();
    state_ = kWorkerStateRunning;
    mtx_->unlock();
    cv_.notify_all();
}

void LaunchWorker::ResumeAndDrop(int64_t drop_idx)
{
    std::unique_lock<MutexLock> lock(*mtx_);

    // Clear all HwCommands.
    sync_cmd_log_.clear();

    for (auto hw_cmd : cmd_log_) hw_cmd->SetState(kCommandStateCompleted);
    cmd_log_.clear();

    drop_idx_ = drop_idx;
    state_ = kWorkerStateRunning;
    
    lock.unlock();
    cv_.notify_all();
}

void LaunchWorker::SyncAll()
{
    std::unique_lock<MutexLock> lock(*mtx_);
    SyncAllWithLock(std::move(lock));
}

void LaunchWorker::SyncCmd(std::shared_ptr<HwCommand> hw_cmd)
{
    std::unique_lock<MutexLock> lock(*mtx_);
    SyncCmdWithLock(std::move(lock), hw_cmd);
}

void LaunchWorker::WorkerLoop()
{
    while (true) {
        XDEBG("worker (%p) waiting for an xcmd", this);
        std::shared_ptr<XCommand> xcmd = kCmdBuf->Dequeue();
        XDEBG("worker (%p) got xcmd (%p)", this, xcmd.get());

        switch (xcmd->GetType())
        {
        case kCommandTypeHardware:
        {
            auto hw_cmd = std::dynamic_pointer_cast<HwCommand>(xcmd);
            XASSERT(hw_cmd != nullptr, "command type mismatch");
            LaunchHwCommand(hw_cmd);
            break;
        }

        case kCommandTypeHostFunction:
        {
            auto cmd = std::dynamic_pointer_cast<HostFunctionCommand>(xcmd);
            XASSERT(cmd != nullptr, "command type mismatch");
            cmd->SetState(kCommandStateInFlight);

            std::unique_lock<MutexLock> lock(*mtx_);
            // Ensure all previous commands have completed.
            lock = SyncAllWithLock(std::move(lock));
            cmd->Execute();
            lock.unlock();

            cmd->SetState(kCommandStateCompleted);
            break;
        }

        case kCommandTypeXQueueWaitAll:
        {
            xcmd->SetState(kCommandStateInFlight);
            SyncAll();
            xcmd->SetState(kCommandStateCompleted);
            break;
        }

        case kCommandTypeBatchSynchronize:
        {
            xcmd->SetState(kCommandStateInFlight);
            std::unique_lock<MutexLock> lock(*mtx_);

            if (sync_cmd_log_.empty()) break;
            auto command = sync_cmd_log_.front();
            lock = SyncCmdWithLock(std::move(lock), command);
            lock.unlock();

            xcmd->SetState(kCommandStateCompleted);
            break;
        }

        case kCommandTypeXQueueDestroy:
        {
            xcmd->SetState(kCommandStateInFlight);
            std::unique_lock<MutexLock> lock(*mtx_);

            lock = SyncAllWithLock(std::move(lock));
            state_ = kWorkerStateTerminated;
            
            lock.unlock();
            cv_.notify_all();

            xcmd->SetState(kCommandStateCompleted);
            // Exit the worker thread.
            return;
        }

        default:
            XASSERT(false, "unknown command type: %d", xcmd->GetType());
            break;
        }
    }
}

void LaunchWorker::LaunchHwCommand(std::shared_ptr<HwCommand> hw_cmd)
{
    XDEBG("launch hw_cmd (%p) idx (%ld)", hw_cmd.get(), hw_cmd->GetIdx());
    // If a HwCommand is deactivated, all non-idempotent HwCommands launched after it should also
    // be deactivated. Otherwise the XPU will become inconsist. So, if a HwCommand is not
    // deactivatable, it should wait until all in-flight deactivatable HwCommands are completed.
    // This is unnecessary under kPreemptLevelBlock.
    bool wait_deactivatable = level_ >= kPreemptLevelDeactivate &&
        (kCommandPropertyNone == hw_cmd->GetProps(
            kCommandPropertyDeactivatable | kCommandPropertyIdempotent));

    hw_cmd->BeforeLaunch();
    std::unique_lock<MutexLock> lock(*mtx_);

    // If the HwCommand is not deactivatable and non-idempoent,
    // wait until all previous in-flight deactivatable HwCommand are completed.
    if (wait_deactivatable) {
        // Find the last synchronizable HwCommand after the last deactivatable HwCommand.
        bool has_deactivatable = false;
        std::shared_ptr<HwCommand> command_to_sync = nullptr;
        for (auto it = cmd_log_.rbegin(); it != cmd_log_.rend(); ++it) {
            if ((*it)->Synchronizable()) command_to_sync = *it;
            if ((*it)->GetProps(kCommandPropertyDeactivatable)) {
                has_deactivatable = true;
                break;
            }
        }

        if (has_deactivatable) {
            // If there is a synchronizable HwCommand, sync it. Otherwise, sync the HwQueue.
            lock = (command_to_sync == nullptr)
                 ? SyncAllWithLock(std::move(lock))
                 : SyncCmdWithLock(std::move(lock), command_to_sync);
        }
    }

    if (cmd_log_.size() >= (size_t)threshold_) {
        // The command log is full (in-flight HwCommand reaches threshold), wait for a empty slot.
        std::shared_ptr<HwCommand> command_to_sync = nullptr;
        int64_t front_command_idx = cmd_log_.front()->GetIdx();

        for (auto cmd : sync_cmd_log_) {
            // Make sure after syncing the HwCommand, there will be at least one empty slot.
            if (cmd->GetIdx() >= front_command_idx) {
                command_to_sync = cmd;
                break;
            }
        }

        // If there is a synchronizable HwCommand, sync it. Otherwise, sync the HwQueue.
        lock = (command_to_sync == nullptr)
             ? SyncAllWithLock(std::move(lock))
             : SyncCmdWithLock(std::move(lock), command_to_sync);

    } else {
        // Wait if the worker is paused.
        while (true) {
            if (state_ == kWorkerStateRunning) {
                break;
            } else if (state_ == kWorkerStatePaused) {
                cv_.wait(lock);
            } else if (state_ == kWorkerStateTerminated) {
                // state_ will not be kWorkerStateTerminated, because LaunchHwCommand() will only
                // be called on worker thread. Right after state_ is set to kWorkerStateTerminated,
                // the thread will then exit.
                XASSERT(false, "Worker state should not be kWorkerStateTerminated.");
            } else {
                XASSERT(false, "Invalid worker state");
            }
        }
    }

    // Check if the HwCommand should be dropped.
    if (hw_cmd->GetIdx() <= drop_idx_) {
        hw_cmd->SetState(kCommandStateCompleted);
        return;
    }

    // Reach the last command in this batch, should enable synchronization for the HwCommand.
    if (hw_cmd->GetIdx() - last_synchronizable_idx_ >= batch_size_) {
        hw_cmd->EnableSynchronization();
    }

    // Launch the HwCommand.
    auto callback = std::dynamic_pointer_cast<HwCallbackCommand>(hw_cmd);
    if (callback != nullptr) {
        XASSERT(callback->Launch(kHwq->GetHandle()) == kXSchedSuccess,
                "failed to launch HwCallbackCommand (%p)", callback.get());
    } else {
        kHwq->Launch(hw_cmd);
    }
    hw_cmd->SetState(kCommandStateInFlight);

    cmd_log_.emplace_back(hw_cmd);

    if (hw_cmd->Synchronizable()) {
        sync_cmd_log_.emplace_back(hw_cmd);
        last_synchronizable_idx_ = hw_cmd->GetIdx();
    }
}

std::unique_lock<MutexLock> LaunchWorker::SyncAllWithLock(std::unique_lock<MutexLock> lock)
{
    while (true) {
        // Wait if the worker is paused.
        while (true) {
            if (state_ == kWorkerStateRunning) {
                break;
            } else if (state_ == kWorkerStatePaused) {
                cv_.wait(lock);
            } else if (state_ == kWorkerStateTerminated) {
                return lock;
            } else {
                XASSERT(false, "Invalid worker state");
            }
        }

        // If there is no in-flight HwCommands, then no need to sync.
        if (cmd_log_.size() == 0) return lock;

        int64_t current_pause_cnt = pause_count_;

        lock.unlock();
        kHwq->Synchronize();
        lock.lock();

        // Check if preemption happened during kHwq->Synchronize().
        if (current_pause_cnt == pause_count_) break;
    }

    // Pop and delete all in-flight HwCommands.
    sync_cmd_log_.clear();

    for (auto hw_cmd : cmd_log_) hw_cmd->SetState(kCommandStateCompleted);
    cmd_log_.clear();

    return lock;
}

std::unique_lock<MutexLock> LaunchWorker::SyncCmdWithLock(std::unique_lock<utils::MutexLock> lock,
                                                          std::shared_ptr<HwCommand> hw_cmd)
{
    XASSERT(hw_cmd->Synchronizable(), "The HwCommand should be synchronizable");

    while (true) {
        // Wait if the worker is paused.
        while (true) {
            // Check if it is already completed.
            // Here checking every loop is necessary because the lock could have been released
            // in the last inner/outer loop and the HwCommand may have been set to
            // kCommandStateCompleted by other threads.
            // It will cause bugs if not checking in the outer loop when the app is syncing this
            // HwCommand, and the XQueue is suspened in the lock-released period in the last
            // outer loop, i.e., unlock(); cmd->sync(); lock();
            // It will also cause bugs if not checking in the inner loop when:
            // 1. the XQueue is suspended
            // 2. this thread is waiting on cv_.wait(lock);
            // 3. lock is released and the XQueue is resumed 
            // 4. hw_cmd is set to completed by another thread
            // 5. the XQueue is again suspended
            // 6. this thread wakes up and locks, finds the XQueue is suspended
            if (hw_cmd->GetState() >= kCommandStateCompleted) return lock;

            if (state_ == kWorkerStateRunning) {
                break;
            } else if (state_ == kWorkerStatePaused) {
                cv_.wait(lock);
            } else if (state_ == kWorkerStateTerminated) {
                return lock;
            } else {
                XASSERT(false, "Invalid worker state");
            }
        }

        XCommandState state = hw_cmd->GetState();
        XASSERT(state >= kCommandStateInFlight, "The syncing HwCommand is not launched");
        if (state == kCommandStateCompleted) break;

        int64_t current_pause_cnt = pause_count_;

        lock.unlock();
        hw_cmd->Synchronize();
        lock.lock();

        // Check if preemption happened during hw_cmd->Synchronize().
        if (current_pause_cnt == pause_count_) break;
    }

    // Pop and delete all HwCommands launched in previous.
    const int64_t current_command_idx = hw_cmd->GetIdx();

    while (sync_cmd_log_.size() > 0) {
        if (sync_cmd_log_.front()->GetIdx() > current_command_idx) break;
        sync_cmd_log_.pop_front();
    }

    while (cmd_log_.size() > 0) {
        auto front_command = cmd_log_.front();
        if (front_command->GetIdx() > current_command_idx) break;
        front_command->SetState(kCommandStateCompleted);
        cmd_log_.pop_front();
    }

    return lock;
}
