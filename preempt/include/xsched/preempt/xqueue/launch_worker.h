#pragma once

#include <list>
#include <thread>
#include <memory>
#include <cstdint>
#include <condition_variable>

#include "xsched/types.h"
#include "xsched/utils/lock.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/preempt/xqueue/command_buffer.h"

namespace xsched::preempt
{

class LaunchWorker
{
public:
    LaunchWorker(std::shared_ptr<HwQueue> hwq, std::shared_ptr<CommandBuffer> buf,
                 XPreemptLevel level, int64_t threshold, int64_t batch_size);
    ~LaunchWorker();

    /// @brief Pause the launch worker.
    void Pause();

    /// @brief Resume the launch worker.
    void Resume();

    /// @brief Resume the launch worker and drop ALL HwCommands.
    /// @param drop_idx Index of the last HwCommand that needs to be dropped.
    /// ANY HwCommands whose idx <= drop_idx will be dropped, EVEN IF they have not been launched.
    void ResumeAndDrop(int64_t drop_idx);

    const CommandLog &GetCommandLog() const { return cmd_log_; }

    /// @brief Synchronize the Worker, will return until
    /// ALL HwCommands submitted to Worker are completed.
    void SyncAll();

    /// @brief Synchronize the HwCommand, will return until the HwCommand is completed.
    /// The HwCommand MUST be Synchronizable and it MUST have been launched
    /// (i.e., GetState() >= kCommandStateInFlight).
    /// @param hw_cmd The Synchronizable HwCommand to synchronize.
    void SyncCmd(std::shared_ptr<HwCommand> hw_cmd);

    void SetPreemptLevel(XPreemptLevel level) { level_ = level; }
    void SetLaunchConfig(int64_t threshold, int64_t batch_size)
    { threshold_ = threshold; batch_size_ = batch_size; }

    int64_t GetThreshold() const { return threshold_; }
    int64_t GetBatchSize() const { return batch_size_; }

private:
    enum WorkerState
    {
        kWorkerStateRunning    = 0,
        kWorkerStatePaused     = 1,
        kWorkerStateTerminated = 2,
    };

    XPreemptLevel level_;
    int64_t threshold_;
    int64_t batch_size_;

    const std::shared_ptr<HwQueue> kHwq = nullptr;
    const std::shared_ptr<CommandBuffer> kCmdBuf = nullptr;

    CommandLog cmd_log_;
    CommandLog sync_cmd_log_;
    int64_t last_synchronizable_idx_ = -1;

    std::condition_variable_any cv_;
    std::unique_ptr<utils::MutexLock> mtx_ = nullptr;

    int64_t drop_idx_ = -1;
    int64_t pause_count_ = 0;
    WorkerState state_ = kWorkerStateRunning;
    std::unique_ptr<std::thread> worker_thread_ = nullptr;

    void WorkerLoop();
    void LaunchHwCommand(std::shared_ptr<HwCommand> hw_cmd);

    std::unique_lock<utils::MutexLock> SyncAllWithLock(std::unique_lock<utils::MutexLock> lock);
    std::unique_lock<utils::MutexLock> SyncCmdWithLock(std::unique_lock<utils::MutexLock> lock,
                                                       std::shared_ptr<HwCommand> hw_cmd);
};

} // namespace xsched::preempt
