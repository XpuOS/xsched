#include "xsched/utils/xassert.h"
#include "xsched/preempt/sched/agent.h"
#include "xsched/preempt/xqueue/command_buffer.h"

using namespace xsched::sched;
using namespace xsched::preempt;

CommandBuffer::CommandBuffer(XQueueHandle xq_h): kXQueueHandle(xq_h)
{
    Enqueue(std::make_shared<XQueueWaitAllCommand>());
    XASSERT(last_enqueued_cmd_ != nullptr, "last_enqueued_cmd_ should not be nullptr");
}

void CommandBuffer::WaitForNextIdle()
{
    std::unique_lock<std::mutex> lock(mtx_);
    if (xq_state_ == kQueueStateIdle) return;
    uint64_t current_idle_cnt = idle_cnt_;
    while (current_idle_cnt == idle_cnt_) idle_cv_.wait(lock);
}

XQueueState CommandBuffer::GetXQueueState()
{
    std::lock_guard<std::mutex> lock(mtx_);
    return xq_state_;
}

void CommandBuffer::ForEachCommand(std::function<bool (std::shared_ptr<XCommand>)> func)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (last_dequeued_cmd_ != nullptr && last_dequeued_cmd_->GetState() < kCommandStateCompleted) {
        func(last_dequeued_cmd_);
    }
    for (auto xcmd : cmds_) {
        if (!func(xcmd)) break;
    }
}

std::shared_ptr<XCommand> CommandBuffer::Dequeue()
{
    std::unique_lock<std::mutex> lock(mtx_);

    if (!cmds_.empty()) {
        auto xcmd = cmds_.front();
        cmds_.pop_front();
        last_dequeued_cmd_ = xcmd;
        XDEBG("xcmd (%p) dequeued from cmd buf (%p)", xcmd.get(), this);
        return xcmd;
    }

    XDEBG("cmd buf (%p) emptied, last_enqueued_cmd_ (%p), type (%d)",
          this, last_enqueued_cmd_.get(), last_enqueued_cmd_->GetType());
    switch (last_enqueued_cmd_->GetType())
    {
    case kCommandTypeXQueueWaitAll:
    {
        if (xq_state_ == kQueueStateReady) {
            xq_state_ = kQueueStateIdle;
            idle_cnt_++;
            idle_cv_.notify_all();
            SchedAgent::SendEvent(std::make_shared<XQueueIdleEvent>(kXQueueHandle));
        }

        while (cmds_.empty()) cv_.wait(lock);
        auto xcmd = cmds_.front();
        cmds_.pop_front();
        last_dequeued_cmd_ = xcmd;
        XDEBG("xcmd (%p) dequeued from cmd buf (%p)", xcmd.get(), this);
        return xcmd;
    }

    case kCommandTypeBatchSynchronize:
    {
        auto xcmd = std::make_shared<XQueueWaitAllCommand>();
        xcmd->SetState(kCommandStatePending);
        last_enqueued_cmd_ = xcmd;
        last_dequeued_cmd_ = xcmd;
        XDEBG("generated XQueueWaitAllCommand (%p) dequeued from cmd buf (%p)", xcmd.get(), this);
        return xcmd;
    }
    
    default:
    {
        auto xcmd = std::make_shared<BatchSynchronizeCommand>();
        xcmd->SetState(kCommandStatePending);
        last_enqueued_cmd_ = xcmd;
        last_dequeued_cmd_ = xcmd;
        XDEBG("generated BatchSynchronizeCommand (%p) dequeued from cmd buf (%p)",
              xcmd.get(), this);
        return xcmd;
    }

    }

    XASSERT(false, "should not reach here");
    return nullptr;
}

void CommandBuffer::Enqueue(std::shared_ptr<XCommand> xcmd)
{
    auto ready_time = std::chrono::system_clock::now();
    XASSERT(xcmd != nullptr, "xcmd should not be nullptr");
    xcmd->SetState(kCommandStatePending);

    mtx_.lock();

    if (xcmd->GetType() == kCommandTypeHardware && xq_state_ == kQueueStateIdle) {
        xq_state_ = kQueueStateReady;
        SchedAgent::SendEvent(std::make_shared<XQueueReadyEvent>(kXQueueHandle, ready_time));
    }

    last_enqueued_cmd_ = xcmd;
    cmds_.emplace_back(xcmd);
    XDEBG("xcmd (%p) type (%d) enqueued to cmd buf (%p)", xcmd.get(), xcmd->GetType(), this);

    mtx_.unlock();
    cv_.notify_all();
}

void CommandBuffer::DropAll()
{
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto &xcmd : cmds_) xcmd->SetState(kCommandStateCompleted);
    cmds_.clear();
}

std::shared_ptr<XQueueWaitAllCommand> CommandBuffer::EnqueueXQueueWaitAllCommand()
{
    std::unique_lock<std::mutex> lock(mtx_);

    if (xq_state_ == kQueueStateIdle) {
        // If the XQueue is idle, syncing on this XQueue should return. Here, we should return
        // a completed XCommand so that syncing on this XQueue will return immediately.
        auto wait_all_cmd = std::dynamic_pointer_cast<XQueueWaitAllCommand>(last_enqueued_cmd_);
        XASSERT(last_enqueued_cmd_->GetType() == kCommandTypeXQueueWaitAll &&
                wait_all_cmd != nullptr,
                "last_enqueued_cmd_ must be an XQueueWaitAllCommand if the XQueue is idle");
        return wait_all_cmd;
    }

    if (last_enqueued_cmd_->GetType() == kCommandTypeXQueueWaitAll) {
        auto wait_all_cmd = std::dynamic_pointer_cast<XQueueWaitAllCommand>(last_enqueued_cmd_);
        XASSERT(wait_all_cmd != nullptr, "last_enqueued_cmd_ must be an XQueueWaitAllCommand");
        return wait_all_cmd;
    }

    auto xcmd = std::make_shared<XQueueWaitAllCommand>();
    xcmd->SetState(kCommandStatePending);
    last_enqueued_cmd_ = xcmd;
    cmds_.emplace_back(xcmd);

    lock.unlock();
    cv_.notify_all();
    return xcmd;
}
