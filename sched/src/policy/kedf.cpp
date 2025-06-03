#include <map>
#include <queue>

#include "xsched/utils/xassert.h"
#include "xsched/sched/policy/kedf.h"

using namespace xsched::sched;


KEarliestDeadlineFirstPolicy::KEarliestDeadlineFirstPolicy()
    : Policy(kPolicyTypeKEarlyDeadlineFirst), K_(1) {
}

typedef std::chrono::system_clock::time_point TimePoint;

struct DeadlineCmp {
    bool operator()(const std::pair<TimePoint, XQueueHandle>& a, const std::pair<TimePoint, XQueueHandle>& b) {
        return a.first > b.first;
    }
};

void KEarliestDeadlineFirstPolicy::Sched(const Status &status)
{
    // find the task with the earliest deadline
    std::priority_queue<std::pair<TimePoint, XQueueHandle>,
                    std::vector<std::pair<TimePoint, XQueueHandle>>,
                    DeadlineCmp> pq;
    auto now = std::chrono::system_clock::now();

    for (auto &status : status.xqueue_status) {
        if (!status.second->ready) continue;

        XQueueHandle handle = status.second->handle;
        auto it = deadlines_.find(handle);
        if (it == deadlines_.end()) continue; // no deadline set

        auto ddl = (status.second->ready_time + std::chrono::microseconds(it->second));
        
        if (ddl < now) {
            // set to a long long time in the future
            ddl = std::chrono::system_clock::now() + std::chrono::hours(1000000);
        }

        auto pair = std::make_pair(ddl, handle);
        pq.push(pair);
    }

    std::unordered_set<XQueueHandle> top_k_handles;
    for (int i = 0; i < K_ && !pq.empty(); ++i) {
        top_k_handles.insert(pq.top().second);
        pq.pop();
    }
    
    // suspend all other xqueues
    for (auto &status : status.xqueue_status) {
        if (top_k_handles.count(status.second->handle)) {
            this->Resume(status.second->handle);
        } else {
            this->Suspend(status.second->handle);
        }
    }
}

void KEarliestDeadlineFirstPolicy::RecvHint(std::shared_ptr<const Hint> hint)
{
    if (hint->Type() == kHintTypeDeadline) {
        auto h = std::dynamic_pointer_cast<const DeadlineHint>(hint);
        XASSERT(h != nullptr, "hint type not match");

        Deadline deadline = h->Ddl();
        deadlines_[h->Handle()] = deadline;
    } else if (hint->Type() == kHintTypeKDeadline) {
        auto h = std::dynamic_pointer_cast<const KDeadlineHint>(hint);
        XASSERT(h != nullptr, "hint type not match");

        K_ = h->K();
    }
}
