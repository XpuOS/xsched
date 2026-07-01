#include <map>

#include "xsched/utils/xassert.h"
#include "xsched/sched/policy/hpf.h"

using namespace xsched::sched;

void HighestPriorityFirstPolicy::Sched(const Status &status)
{
    // find the running highest priority task of each device.
    // Skip XQueues whose ready heartbeat has expired (no Ready event for >5s).
    auto now = std::chrono::system_clock::now();
    std::map<XDevice, Priority> running_prio_max;
    for (auto &status : status.xqueue_status) {
        if (!status.second->ready) continue;
        if (now - status.second->ready_time > std::chrono::seconds(5)) continue;
        XQueueHandle handle = status.second->handle;
        Priority priority = GetPriority(handle);

        auto prio_it = running_prio_max.find(status.second->device);
        if (prio_it == running_prio_max.end()) {
            running_prio_max[status.second->device] = priority;
        } else if (priority > prio_it->second) {
            prio_it->second = priority;
        }
    }

    // suspend all xqueues with lower priority
    // and resume all xqueues with higher priority
    for (auto &status : status.xqueue_status) {
        XQueueHandle handle = status.second->handle;
        Priority priority = GetPriority(handle);

        // Stale ready → suspend it
        if (status.second->ready &&
            now - status.second->ready_time > std::chrono::seconds(5)) {
            this->Suspend(handle);
            continue;
        }

        // get the running highest priority task of the device
        Priority prio_max = PRIORITY_MIN;
        auto prio_it = running_prio_max.find(status.second->device);
        if (prio_it != running_prio_max.end()) prio_max = prio_it->second;

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
        XWARN("priority %d not in range [%d, %d], overide priority for XQueue 0x" FMT_64X " to %d",
              h->Prio(), PRIORITY_MIN, PRIORITY_MAX, h->Handle(), priority);
    }

    XINFO("set priority %d for XQueue 0x" FMT_64X, priority, h->Handle());
    priorities_[h->Handle()] = priority;
}

Priority HighestPriorityFirstPolicy::GetPriority(XQueueHandle handle)
{
    auto it = priorities_.find(handle);
    if (it != priorities_.end()) return it->second;
    // if priority not found, use default priority
    return PRIORITY_DEFAULT;
}
