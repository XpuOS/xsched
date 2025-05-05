#include <map>

#include "xsched/utils/xassert.h"
#include "xsched/sched/policy/edf.h"

using namespace xsched::sched;

void EarliestDeadlineFirstPolicy::Sched(const Status &status)
{
    // find the task with the earliest deadline
    XQueueHandle earliest_xqueue = 0;
    auto earliest_deadline = std::chrono::system_clock::time_point::max();
    for (auto &status : status.xqueue_status) {
        if (!status.second->ready) continue;

        XQueueHandle handle = status.second->handle;
        auto it = deadlines_.find(handle);
        if (it == deadlines_.end()) continue; // no deadline set

        auto ddl = status.second->ready_time + std::chrono::seconds(it->second);
        if (ddl >= earliest_deadline) continue;

        earliest_deadline = ddl;
        earliest_xqueue = handle;
    }

    // suspend all other xqueues
    for (auto &status : status.xqueue_status) {
        if (status.second->handle == earliest_xqueue) {
            this->Resume(status.second->handle);
        } else {
            this->Suspend(status.second->handle);
        }
    }
}

void EarliestDeadlineFirstPolicy::RecvHint(std::shared_ptr<const Hint> hint)
{
    if (hint->Type() != kHintTypeDeadline) return;
    auto h = std::dynamic_pointer_cast<const DeadlineHint>(hint);
    XASSERT(h != nullptr, "hint type not match");

    Deadline deadline = h->Ddl();
    deadlines_[h->Handle()] = deadline;
}
