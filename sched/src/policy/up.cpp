#include "xsched/utils/xassert.h"
#include "xsched/sched/policy/up.h"

using namespace xsched::sched;

void UtilizationPartitionPolicy::Sched(const Status &status)
{
    for (auto it = utils_.begin(); it != utils_.end();) {
        PID pid = it->first;
        if (status.process_status.find(pid) == status.process_status.end()) {
            it = utils_.erase(it);
        } else {
            ++it;
        }
    }

    for (const auto &xq : status.xqueue_status) {
        handle_to_pid_[xq.first] = xq.second->pid;
    }

    if (cur_running_ != 0 &&
        status.process_status.find(cur_running_) == status.process_status.end()) {
        cur_running_ = 0;
    }

    if (utils_.empty()) return;

    if (cur_running_ == 0) {
        SwitchToAny(status);
        return;
    }

    auto now = std::chrono::system_clock::now();
    if (now < cur_end_ && ProcessReady(cur_running_, status)) return;

    auto bit = utils_.find(cur_running_);
    if (bit == utils_.end()) {
        SwitchToAny(status);
        return;
    }

    for (++bit; bit != utils_.end(); ++bit) {
        if (!ProcessReady(bit->first, status)) continue;
        SwitchProcess(bit->first, bit->second, status);
        return;
    }

    for (bit = utils_.begin(); bit != utils_.end(); ++bit) {
        if (bit->first == cur_running_) break;
        if (!ProcessReady(bit->first, status)) continue;
        SwitchProcess(bit->first, bit->second, status);
        return;
    }

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
        PID pid = h->Pid();
        if (pid != 0) {
            utils_[pid] = util;
        } else {
            auto it = handle_to_pid_.find(h->Handle());
            if (it != handle_to_pid_.end()) {
                utils_[it->second] = util;
            }
        }
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
    for (const auto &process : utils_) { total_util += process.second; }
    if(total_util == 0) return std::chrono::microseconds(TIMESLICE_DEFAULT);
    return std::chrono::microseconds(totalUs * util / total_util);
}

bool UtilizationPartitionPolicy::ProcessReady(PID pid, const Status &status)
{
    auto it = status.process_status.find(pid);
    if (it == status.process_status.end()) return false;
    for (auto xq : it->second->running_xqueues) {
        auto xit = status.xqueue_status.find(xq);
        if (xit != status.xqueue_status.end() && xit->second->ready) return true;
    }
    for (auto xq : it->second->suspended_xqueues) {
        auto xit = status.xqueue_status.find(xq);
        if (xit != status.xqueue_status.end() && xit->second->ready) return true;
    }
    return false;
}

void UtilizationPartitionPolicy::SwitchToAny(const Status &status)
{
    cur_running_ = 0;
    for (const auto &process : status.process_status) {
        PID pid = process.first;
        if (!ProcessReady(pid, status)) {
            for (const auto xq : process.second->running_xqueues)
                this->Suspend(xq);
            for (const auto xq : process.second->suspended_xqueues)
                this->Suspend(xq);
            continue;
        }

        auto it = utils_.find(pid);
        if (it == utils_.end()) {
            for (const auto xq : process.second->running_xqueues)
                this->Suspend(xq);
            for (const auto xq : process.second->suspended_xqueues)
                this->Suspend(xq);
            continue;
        }

        SwitchProcess(pid, it->second, status);
        return;
    }
}

void
UtilizationPartitionPolicy::SwitchProcess(PID pid, Utilization util, const Status &status)
{
    for (const auto &process : status.process_status) {
        const auto it = status.process_status.find(process.first);
        if (it == status.process_status.end()) continue;

        if (process.first == pid) {
            std::list<XQueueHandle> suspended_xqueues;
            for (const auto xq : it->second->suspended_xqueues)
                suspended_xqueues.push_back(xq);
            for (const auto xq : suspended_xqueues) { this->Resume(xq); }
        } else {
            std::list<XQueueHandle> running_xqueues;
            for (const auto xq : it->second->running_xqueues)
                running_xqueues.push_back(xq);
            for (const auto xq : running_xqueues) { this->Suspend(xq); }
        }
    }

    cur_running_ = pid;
    cur_end_ = std::chrono::system_clock::now() + GetBudget(util);
    this->AddTimer(cur_end_);
}
