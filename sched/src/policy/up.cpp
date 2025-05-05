#include "xsched/utils/xassert.h"
#include "xsched/sched/policy/up.h"

using namespace xsched::sched;

void UtilizationPartitionPolicy::Sched(const Status &status)
{
    // delete all destroyed xqueues and their utilizations
    for (auto it = utils_.begin(); it != utils_.end();) {
        XQueueHandle xqueue = it->first;
        if (status.xqueue_status.find(xqueue) == status.xqueue_status.end()) {
            it = utils_.erase(it);
        } else {
            ++it;
        }
    }

    // delete destroyed current running xqueue
    if (cur_running_ != 0 &&
        status.xqueue_status.find(cur_running_) == status.xqueue_status.end()) {
        cur_running_ = 0;
    }

    if (utils_.empty()) return;

    if (cur_running_ == 0) {
        // nothing is running
        SwitchToAny(status);
        return;
    }

    auto xit = status.xqueue_status.find(cur_running_);
    if (xit == status.xqueue_status.end()) {
        // current xqueue is not found
        auto bit = utils_.find(cur_running_);
        XASSERT(bit != utils_.end(), "utilization of XQueue 0x%lx not found.", cur_running_);
        utils_.erase(bit);
        SwitchToAny(status);
        return;
    }

    auto now = std::chrono::system_clock::now();
    if (now < cur_end_ && xit->second->ready) return;

    auto bit = utils_.find(cur_running_);
    XASSERT(bit != utils_.end(), "utilization of XQueue 0x%lx not found.", cur_running_);
    
    // current xqueue has finished its time slice
    // select the next xqueue to run
    for (++bit; bit != utils_.end();) {
        auto xit = status.xqueue_status.find(bit->first);
        if (xit == status.xqueue_status.end()) {
            // xqueue not found
            utils_.erase(bit++);
            continue;
        }
        if (!xit->second->ready) {
            // not running check the next one
            ++bit;
            continue;
        }
        SwitchTo(bit->first, bit->second, status);
        return;
    }

    for (bit = utils_.begin(); bit != utils_.end();) {
        if (bit->first == cur_running_) break;
        
        auto xit = status.xqueue_status.find(bit->first);
        if (xit == status.xqueue_status.end()) {
            // xqueue not found
            utils_.erase(bit++);
            continue;
        }
        if (!xit->second->ready) {
            // not running check the next one
            ++bit;
            continue;
        }
        SwitchTo(bit->first, bit->second, status);
        return;
    }

    // checked a round, no xqueue to run
    cur_running_ = 0;
}

void UtilizationPartitionPolicy::RecvHint(std::shared_ptr<const Hint> hint)
{
    switch (hint->Type())
    {
    case kHintTypeUtilization:
    {
        auto h = std::dynamic_pointer_cast<const UtilizationHint>(hint);
        XASSERT(h != nullptr, "hint type not match");
        Utilization util = h->Util();
        if (util < UTILIZATION_MIN || util > UTILIZATION_MAX) {
            XWARN("invalid utilization %d", util);
            break;
        }
        utils_[h->Handle()] = h->Util();
        break;
    }
    case kHintTypeTimeslice:
    {
        auto h = std::dynamic_pointer_cast<const TimesliceHint>(hint);
        XASSERT(h != nullptr, "hint type not match");
        timeslice_ = std::chrono::microseconds(h->Ts());
        break;
    }
    default:
        XWARN("unsupported hint type: %d", hint->Type());
        break;
    }
}

std::chrono::microseconds UtilizationPartitionPolicy::GetBudget(Utilization util)
{
    Utilization total_util = 0;
    int64_t totalUs = timeslice_.count();
    for (const auto &xqueue : utils_) { total_util += xqueue.second; }
    return std::chrono::microseconds(totalUs * util / total_util);
}

void UtilizationPartitionPolicy::SwitchToAny(const Status &status)
{
    cur_running_ = 0;
    bool selected = false;
    for (const auto &xqueue : status.xqueue_status) {
        if (selected || !xqueue.second->ready) {
            this->Suspend(xqueue.first);
            continue;
        }

        auto it = utils_.find(xqueue.first);
        if (it == utils_.end()) {
            this->Suspend(xqueue.first);
            continue;
        }

        selected = true;
        this->Resume(xqueue.first);
        cur_running_ = xqueue.first;
        cur_end_ = std::chrono::system_clock::now() + GetBudget(it->second);
        this->AddTimer(cur_end_);
    }
}

void
UtilizationPartitionPolicy::SwitchTo(XQueueHandle handle, Utilization util, const Status &status)
{
    for (const auto &xqueue : status.xqueue_status) {
        if (xqueue.first == handle) continue;
        this->Suspend(xqueue.first);
    }

    this->Resume(handle);
    cur_running_ = handle;
    cur_end_ = std::chrono::system_clock::now() + GetBudget(util);
    this->AddTimer(cur_end_);
}
