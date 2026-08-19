#include <map>
#include <list>

#include "xsched/utils/xassert.h"
#include "xsched/sched/policy/lax.h"

using namespace xsched::sched;

void LaxityPolicy::Sched(const Status &status)
{
    auto current = std::chrono::system_clock::now();

    for (const auto &xq : status.xqueue_status) {
        handle_to_pid_[xq.first] = xq.second->pid;
    }

    Priority highest_prio = PRIORITY_MIN;
    bool has_laxity = false;
    auto earliest_laxity = (std::chrono::system_clock::time_point::max)();

    for (const auto &process : status.process_status) {
        PID pid = process.first;

        auto earliest_ready = (std::chrono::system_clock::time_point::max)();
        bool any_ready = false;
        for (auto handle : process.second->running_xqueues) {
            auto it = status.xqueue_status.find(handle);
            if (it == status.xqueue_status.end()) continue;
            if (it->second->ready) {
                any_ready = true;
                if (it->second->ready_time < earliest_ready)
                    earliest_ready = it->second->ready_time;
            }
        }
        for (auto handle : process.second->suspended_xqueues) {
            auto it = status.xqueue_status.find(handle);
            if (it == status.xqueue_status.end()) continue;
            if (it->second->ready) {
                any_ready = true;
                if (it->second->ready_time < earliest_ready)
                    earliest_ready = it->second->ready_time;
            }
        }
        if (!any_ready) continue;

        auto it = laxity_infos_.find(pid);
        if (it == laxity_infos_.end()) {
            if (PRIORITY_DEFAULT > highest_prio) highest_prio = PRIORITY_DEFAULT;
            continue;
        }

        auto laxity = earliest_ready + std::chrono::microseconds(it->second.lax);
        Priority prio = current < laxity ? it->second.lax_prio : it->second.crit_prio;

        if (prio > highest_prio) highest_prio = prio;
        if (current < laxity && laxity < earliest_laxity) {
            earliest_laxity = laxity;
            has_laxity = true;
        }
    }

    for (const auto &process : status.process_status) {
        PID pid = process.first;

        auto earliest_ready = (std::chrono::system_clock::time_point::max)();
        bool any_ready = false;
        for (auto handle : process.second->running_xqueues) {
            auto it = status.xqueue_status.find(handle);
            if (it == status.xqueue_status.end()) continue;
            if (it->second->ready) {
                any_ready = true;
                if (it->second->ready_time < earliest_ready)
                    earliest_ready = it->second->ready_time;
            }
        }
        for (auto handle : process.second->suspended_xqueues) {
            auto it = status.xqueue_status.find(handle);
            if (it == status.xqueue_status.end()) continue;
            if (it->second->ready) {
                any_ready = true;
                if (it->second->ready_time < earliest_ready)
                    earliest_ready = it->second->ready_time;
            }
        }

        Priority prio = PRIORITY_DEFAULT;
        auto lit = laxity_infos_.find(pid);
        if (lit != laxity_infos_.end() && any_ready) {
            auto laxity = earliest_ready + std::chrono::microseconds(lit->second.lax);
            prio = current < laxity ? lit->second.lax_prio : lit->second.crit_prio;
        }

        if (prio < highest_prio) {
            std::list<XQueueHandle> running;
            for (const auto &xq : process.second->running_xqueues)
                running.push_back(xq);
            for (const auto xq : running) this->Suspend(xq);
        } else {
            std::list<XQueueHandle> suspended;
            for (const auto &xq : process.second->suspended_xqueues)
                suspended.push_back(xq);
            for (const auto xq : suspended) this->Resume(xq);
        }
    }

    if (has_laxity) {
        this->AddTimer(earliest_laxity);
    }
}

void LaxityPolicy::RecvHint(std::shared_ptr<const Hint> hint)
{
    if (hint->Type() != kHintTypeLaxity) return;
    auto h = std::dynamic_pointer_cast<const LaxityHint>(hint);
    XASSERT(h != nullptr, "hint type not match");

    Laxity lax = h->Lax();
    Priority lax_prio = h->LaxPrio();
    Priority crit_prio = h->CritPrio();

    auto pid_it = handle_to_pid_.find(h->Handle());
    PID pid = (pid_it != handle_to_pid_.end()) ? pid_it->second : (PID)h->Handle();
    laxity_infos_[pid] = {
        .lax = lax < 0 ? NO_LAXITY : lax,
        .lax_prio = lax_prio,
        .crit_prio = crit_prio
    };
}
