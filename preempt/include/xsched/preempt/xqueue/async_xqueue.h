#pragma once

#include <memory>
#include <atomic>

#include "xsched/protocol/def.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/preempt/xqueue/launch_worker.h"
#include "xsched/preempt/xqueue/command_buffer.h"

namespace xsched::preempt
{

class AsyncXQueue : public XQueue
{
public:
    AsyncXQueue(std::shared_ptr<HwQueue> hwq, XPreemptLevel level,
                int64_t threshold  = XSCHED_DEFAULT_COMMAND_THRESHOLD,
                int64_t batch_size = XSCHED_DEFAULT_COMMAND_BATCH_SZIE);
    virtual ~AsyncXQueue();

    virtual void Submit(std::shared_ptr<HwCommand> hw_cmd) override;
    virtual std::shared_ptr<XQueueWaitAllCommand> SubmitWaitAll() override;
    virtual void WaitAll() override;
    virtual void Wait(std::shared_ptr<HwCommand> hw_cmd) override;
    virtual XQueueState Query() override;
    virtual int64_t GetHwCommandCount() override;

    virtual void Suspend(int64_t flags) override;
    virtual void Resume(int64_t flags) override;

    virtual void SetPreemptLevel(XPreemptLevel level) override;
    virtual void SetLaunchConfig(int64_t threshold, int64_t batch_size) override;

private:
    XPreemptLevel level_;
    const std::shared_ptr<CommandBuffer> cmd_buf_ = nullptr;
    const std::shared_ptr<LaunchWorker> launch_worker_ = nullptr;

    std::atomic_bool suspended_ = { false };
    std::atomic_bool terminated_ = { false };

    // The index of HwCommand starts from 1.
    std::atomic<int64_t> next_hw_cmd_idx_ = { 1 };
};

} // namespace xsched::preempt
