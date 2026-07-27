#include <map>
#include <vector>
#include <algorithm>

#include "xsched/utils/xassert.h"
#include "xsched/sched/policy/kedf.h"

using namespace xsched::sched;

void KEarliestDeadlineFirstPolicy::Sched(const Status &status)
{
    std::vector<ProcessDeadlineEntry> ddls;
    ddls.reserve(status.process_status.size());

    for (auto &process : status.process_status) {
        PID pid = process.first;
        auto ddl = (std::chrono::system_clock::time_point::max)();

        bool any_ready = false;
        for (auto handle : process.second->running_xqueues) {
            auto xq_it = status.xqueue_status.find(handle);
            if (xq_it == status.xqueue_status.end()) continue;
            if (!xq_it->second->ready) continue;
            any_ready = true;
            auto d_it = deadlines_.find(handle);
            if (d_it == deadlines_.end()) continue;
            auto xq_ddl = xq_it->second->ready_time + std::chrono::microseconds(d_it->second);
            if (xq_ddl < ddl) ddl = xq_ddl;
        }
        for (auto handle : process.second->suspended_xqueues) {
            auto xq_it = status.xqueue_status.find(handle);
            if (xq_it == status.xqueue_status.end()) continue;
            if (!xq_it->second->ready) continue;
            any_ready = true;
            auto d_it = deadlines_.find(handle);
            if (d_it == deadlines_.end()) continue;
            auto xq_ddl = xq_it->second->ready_time + std::chrono::microseconds(d_it->second);
            if (xq_ddl < ddl) ddl = xq_ddl;
        }

        if (!any_ready) {
            ddls.emplace_back(ProcessDeadlineEntry{.pid=pid,.deadline=(std::chrono::system_clock::time_point::max)()});
            continue;
        }

        ddls.emplace_back(ProcessDeadlineEntry{.pid=pid,.deadline=ddl});
    }

    std::sort(ddls.begin(), ddls.end(), [](const ProcessDeadlineEntry &a, const ProcessDeadlineEntry &b) {
        return a.deadline < b.deadline;
    });

    for (size_t i = 0; i < k_ && i < ddls.size(); ++i) {
        SwitchProcess(ddls[i].pid, status);
    }
    for (size_t i = k_; i < ddls.size(); ++i) {
        const auto it = status.process_status.find(ddls[i].pid);
        if (it == status.process_status.end()) continue;
        std::vector<XQueueHandle> running;
        for (auto xq : it->second->running_xqueues) running.push_back(xq);
        for (auto xq : running) this->Suspend(xq);
    }
}

void KEarliestDeadlineFirstPolicy::SwitchProcess(PID pid, const Status &status)
{
    const auto it = status.process_status.find(pid);
    if (it == status.process_status.end()) return;

    std::vector<XQueueHandle> suspended;
    for (auto xq : it->second->suspended_xqueues) suspended.push_back(xq);
    for (auto xq : suspended) this->Resume(xq);
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
            XWARN("invalid k " FMT_64U, k);
            break;
        }
        k_ = k;
        XINFO("k set to " FMT_64U, k);
        break;
    }
    default:
        XWARN("unsupported hint type: %d", hint->Type());
        break;
    }
}
