#include "xsched/utils/xassert.h"
#include "xsched/sched/policy/awf.h"

using namespace xsched::sched;

void ActiveWindowFirstPolicy::Sched(const Status &status)
{
    // check if any active processes have any ready XQueues
    bool active_process_ready = false;
    for (auto &xq : status.xqueue_status) {
        if (!xq.second->ready) continue; // skip if not ready
        if (active_pids_.count(xq.second->pid)) {
            active_process_ready = true;
            break;
        }
    }

    // if there are no ready XQueues in any active processes, resume all XQueues
    if (!active_process_ready) {
        for (auto &xq : status.xqueue_status) Resume(xq.first);
        return;
    }

    // suspend all XQueues except the active processes
    for (auto &xq : status.xqueue_status) {
        if (active_pids_.count(xq.second->pid)) Resume(xq.first);
        else Suspend(xq.first);
    }
}

void ActiveWindowFirstPolicy::RecvHint(std::shared_ptr<const Hint> hint)
{
    if (hint->Type() != kHintTypeWindowActive) return;
    auto wa_hint = std::dynamic_pointer_cast<const WindowActiveHint>(hint);
    XASSERT(wa_hint != nullptr, "hint type not match");
    PID pid = wa_hint->Pid();
    if (pid == 0) return;

    // one display can only have one active window (pid)
    uint64_t display = wa_hint->Display();
    auto it = display_active_pids_.find(display);
    if (it == display_active_pids_.end()) {
        active_pids_.insert(pid);
        display_active_pids_[display] = pid;
        return;
    }

    active_pids_.erase(it->second);
    active_pids_.insert(pid);
    display_active_pids_[display] = pid;
}
