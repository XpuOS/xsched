#include <map>

#include "xsched/utils/xassert.h"
#include "xsched/sched/policy/hhpf.h"

using namespace xsched::sched;

void HeterogeneousHighestPriorityFirstPolicy::Sched(const Status &status)
{
    // find the running highest priority task in the system
    Priority prio_max = PRIORITY_MIN;
    for (auto &status : status.xqueue_status) {
        if (!status.second->ready) continue;
        XQueueHandle handle = status.second->handle;
        Priority priority = GetPriority(handle);
        if (priority > prio_max) prio_max = priority;
    }

    // suspend all xqueues with lower priority
    // and resume all xqueues with higher priority
    for (auto &status : status.xqueue_status) {
        XQueueHandle handle = status.second->handle;
        Priority priority = GetPriority(handle);
        if (priority < prio_max) {
            this->Suspend(handle);
        } else {
            this->Resume(handle);
        }
    }
}

void HeterogeneousHighestPriorityFirstPolicy::RecvHint(std::shared_ptr<const Hint> hint)
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

Priority HeterogeneousHighestPriorityFirstPolicy::GetPriority(XQueueHandle handle)
{
    auto it = priorities_.find(handle);
    if (it != priorities_.end()) return it->second;
    // if priority not found, use default priority
    return PRIORITY_DEFAULT;
}
