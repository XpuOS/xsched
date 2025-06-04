#include <map>
#include <vector>
#include <algorithm>

#include "xsched/utils/xassert.h"
#include "xsched/sched/policy/kedf.h"

using namespace xsched::sched;

void KEarliestDeadlineFirstPolicy::Sched(const Status &status)
{
    std::vector<DeadlineEntry> ddls;
    ddls.reserve(status.xqueue_status.size());

    // calculate the deadline of each xqueue
    for (auto &status : status.xqueue_status) {
        XQueueHandle handle = status.second->handle;
        auto ddl = std::chrono::system_clock::time_point::max();
        if (!status.second->ready) {
            ddls.emplace_back(DeadlineEntry{.xqueue=handle,.deadline=ddl});
            continue;
        }

        auto it = deadlines_.find(handle);
        if (it == deadlines_.end()) {
            ddls.emplace_back(DeadlineEntry{.xqueue=handle,.deadline=ddl});
            continue;
        }

        ddl = status.second->ready_time + std::chrono::microseconds(it->second);
        ddls.emplace_back(DeadlineEntry{.xqueue=handle,.deadline=ddl});
    }

    // sort the xqueues by deadline, from the earliest to the latest
    std::sort(ddls.begin(), ddls.end(), [](const DeadlineEntry &a, const DeadlineEntry &b) {
        return a.deadline < b.deadline;
    });

    // resume the first k_ xqueues
    for (size_t i = 0; i < k_ && i < ddls.size(); ++i) {
        this->Resume(ddls[i].xqueue);
    }

    // suspend all other xqueues
    for (size_t i = k_; i < ddls.size(); ++i) {
        this->Suspend(ddls[i].xqueue);
    }
}

void KEarliestDeadlineFirstPolicy::RecvHint(std::shared_ptr<const Hint> hint)
{
    switch (hint->Type())
    {
    case kHintTypeDeadline:
    {
        auto h = std::dynamic_pointer_cast<const DeadlineHint>(hint);
        XASSERT(h != nullptr, "hint type not match");
        deadlines_[h->Handle()] = h->Ddl();
        break;
    }
    case kHintTypeKDeadline:
    {
        auto h = std::dynamic_pointer_cast<const KDeadlineHint>(hint);
        XASSERT(h != nullptr, "hint type not match");
        size_t k = h->K();
        if (k < 1) {
            XWARN("invalid k %lu", k);
            break;
        }
        k_ = k;
        XINFO("k set to %lu", k);
        break;
    }
    default:
        XWARN("unsupported hint type: %d", hint->Type());
        break;
    }
}
