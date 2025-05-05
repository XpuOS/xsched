#include "xsched/utils/xassert.h"
#include "xsched/preempt/xqueue/xcommand.h"

using namespace xsched::preempt;

XCommandState XCommand::GetState()
{
    std::lock_guard<std::mutex> lock(mtx_);
    return state_;
}

void XCommand::SetState(XCommandState new_state)
{
    XASSERT(new_state >= kCommandStateCreated && new_state <= kCommandStateCompleted,
            "invalid state: %d", new_state);
    mtx_.lock();
    XCommandState old_state = state_;
    XASSERT(new_state >= old_state, "state should not go back");
    state_ = new_state;
    mtx_.unlock();

    if (new_state == old_state) return; // only notify when state changes
    state_cv_.notify_all();
    for (int i = old_state + 1; i <= new_state; i++) {
        for (auto listener : state_listeners_) listener(XCommandState(i));
    }
}

void XCommand::WaitUntil(XCommandState state)
{
    std::unique_lock<std::mutex> lock(mtx_);
    while ((int)state_ < (int)state) state_cv_.wait(lock);
}
