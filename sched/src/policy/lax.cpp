#include <map>

#include "xsched/utils/xassert.h"
#include "xsched/sched/policy/lax.h"

using namespace xsched::sched;

void LaxityPolicy::Sched(const Status &status)
{
    auto current = std::chrono::system_clock::now();

    // find the task with the highest priority and the earliest laxity
    Priority highest_prio = PRIORITY_MIN;
    bool has_laxity = false;
    auto earliest_laxity = std::chrono::system_clock::time_point::max();
    for (auto &status : status.xqueue_status) {
        if (!status.second->ready) continue;

        XQueueHandle handle = status.second->handle;
        auto it = laxity_infos_.find(handle);
        if (it == laxity_infos_.end()) {
            // no laxity info set, use PRIO_DEFAULT
            if (PRIORITY_DEFAULT > highest_prio) highest_prio = PRIORITY_DEFAULT;
            continue;
        }

        auto laxity = status.second->ready_time + std::chrono::microseconds(it->second.lax);
        Priority prio = current < laxity ? it->second.lax_prio : it->second.crit_prio;

        if (prio > highest_prio) highest_prio = prio;
        if (current < laxity && laxity < earliest_laxity) {
            earliest_laxity = laxity;
            has_laxity = true;
        }
    }

    // suspend all other xqueues
    for (auto &status : status.xqueue_status) {
        Priority prio = PRIORITY_DEFAULT;
        XQueueHandle handle = status.second->handle;
        auto it = laxity_infos_.find(handle);
        if (it != laxity_infos_.end()) {
            auto laxity = status.second->ready_time + std::chrono::microseconds(it->second.lax);
            prio = !status.second->ready || current < laxity
                 ? it->second.lax_prio
                 : it->second.crit_prio;
        }

        if (prio < highest_prio) this->Suspend(handle);
        else this->Resume(handle);
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
    laxity_infos_[h->Handle()] = {
        .lax = lax < 0 ? NO_LAXITY : lax,
        .lax_prio = lax_prio,
        .crit_prio = crit_prio
    };
}
