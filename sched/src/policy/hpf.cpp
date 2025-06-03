#include <map>

#include "xsched/utils/xassert.h"
#include "xsched/sched/policy/hpf.h"

using namespace xsched::sched;

HighestPriorityFirstPolicy::HighestPriorityFirstPolicy(): Policy(kPolicyTypeHighestPriorityFirst)
{
    mode_ = kModeDefault;
    const char *mode_str = std::getenv("XSCHED_HPF_MODE");
    if (mode_str) {
        if (strcmp(mode_str, "cosched") == 0) {
            mode_ = kModeCosched;
            XINFO("HPF mode: cosched");
        } else {
            XWARN("invalid value of XSCHED_HPF_MODE: %s, "
                  "valid values: [default, cosched]", mode_str);
            mode_ = kModeDefault;
        }
    }
}

void HighestPriorityFirstPolicy::Sched(const Status &status)
{
    // find the highest priority task of each device
    std::map<XDevice, Priority> running_prio_max;
    for (auto &status : status.xqueue_status) {
        XQueueHandle handle = status.second->handle;
        Priority priority = PRIORITY_DEFAULT;
        auto it = priorities_.find(handle);
        // if priority not found, use default priority
        if (it == priorities_.end()) priorities_[handle] = priority;
        else priority = it->second;

        if (!status.second->ready) continue;
        auto prio_it = running_prio_max.find(status.second->device);
        if (prio_it == running_prio_max.end()) {
            running_prio_max[status.second->device] = priority;
        } else if (priority > prio_it->second) {
            prio_it->second = priority;
        }
    }

    // For co-sched mode, we set the highest priority of each device
    // to the same value (the highest priority of all devices).
    // So that, all devices can preempt each other.
    if (mode_ == kModeCosched) {
        Priority prio_max = PRIORITY_MIN;
        for (auto &prio_it : running_prio_max) {
            if (prio_it.second > prio_max) {
                prio_max = prio_it.second;
            }
        }
        for (auto &prio_it : running_prio_max) {
            prio_it.second = prio_max;
        }
    }

    // suspend all xqueues with lower priority
    // and resume all xqueues with higher priority
    for (auto &status : status.xqueue_status) {
        XQueueHandle handle = status.second->handle;
        if (!status.second->ready) continue;
        auto it = priorities_.find(handle);
        XASSERT(it != priorities_.end(), "priority of XQueue 0x%lx not found.", handle);
        Priority priority = it->second;

        Priority prio_max = PRIORITY_MIN;
        auto prio_it = running_prio_max.find(status.second->device);
        if (prio_it != running_prio_max.end()) {
            prio_max = prio_it->second;
        }

        if (priority < prio_max) {
            this->Suspend(handle);
        } else {
            this->Resume(handle);
        }
    }
}

void HighestPriorityFirstPolicy::RecvHint(std::shared_ptr<const Hint> hint)
{
    if (hint->Type() != kHintTypePriority) return;
    auto h = std::dynamic_pointer_cast<const PriorityHint>(hint);
    XASSERT(h != nullptr, "hint type not match");

    Priority priority = h->Prio();
    if (priority < PRIORITY_MIN) priority = PRIORITY_MIN;
    if (priority > PRIORITY_MAX) priority = PRIORITY_MAX;
    if (priority != h->Prio()) {
        XWARN("priority %d for XQueue 0x%lu is invalid, "
              "valid range: [%d, %d], priority overide to %d",
              h->Prio(), h->Handle(), PRIORITY_MIN, PRIORITY_MAX, priority);
    }

    XINFO("set priority %d for XQueue 0x%lx", priority, h->Handle());
    priorities_[h->Handle()] = priority;
}
