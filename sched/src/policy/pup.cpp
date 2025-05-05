#include "xsched/utils/xassert.h"
#include "xsched/sched/policy/pup.h"

using namespace std::chrono;
using namespace xsched::sched;

void ProcessUtilizationPartitionPolicy::Sched(const Status &status)
{
    auto now = system_clock::now();

    // delete all destroyed pids and their utilizations
    for (auto it = utils_.begin(); it != utils_.end();) {
        PID pid = it->first;
        if (status.process_status.find(pid) == status.process_status.end()) {
            it = utils_.erase(it);
        } else {
            ++it;
        }
    }

    // check if any process is ready
    bool any_ready = false;

    // delete all destroyed processes and processes with no XQueues from the run queue
    for (auto it = run_queue_.begin(); it != run_queue_.end();) {
        PID pid = it->pid;
        auto ps_it = status.process_status.find(pid);
        if (ps_it == status.process_status.end() ||
           (ps_it->second->running_xqueues.empty() && ps_it->second->suspended_xqueues.empty())) {
            it = run_queue_.erase(it);
            pid_in_queue_.erase(pid);
        } else {
            bool ready = ProcessReady(pid, status);
            it->ready = ready;
            any_ready = any_ready || ready;
            ++it;
        }
    }

    // add new processes with XQueues to the run queue
    for (const auto &process : status.process_status) {
        PID pid = process.first;
        if (process.second->running_xqueues.empty() && process.second->suspended_xqueues.empty()) {
            continue;
        }
        if (pid_in_queue_.find(pid) != pid_in_queue_.end()) continue;
        bool ready = ProcessReady(pid, status);
        any_ready = any_ready || ready;
        run_queue_.emplace_back(RunEntry{.pid=pid,.ready=ready,.running=false,.budget_us=0});
        pid_in_queue_.insert(pid);
    }

    // nothing to run
    if (run_queue_.empty()) return;
    if (!any_ready) return;

    if (run_queue_.front().running) {
        // if the timeslice has not ended and the process is still running, do nothing
        bool ts_remain = timeslice_end_ > now;
        bool ready = run_queue_.front().ready;
        if (ts_remain && ready) return;

        // move the first process to the end of the queue
        PID pid = run_queue_.front().pid;
        run_queue_.pop_front();

        // save unused timeslice as its budget
        int64_t budget = ts_remain ? duration_cast<microseconds>(timeslice_end_ - now).count() : 0;
        run_queue_.emplace_back(RunEntry{.pid=pid,.ready=ready,.running=false,.budget_us=budget});
    }

    // find the first ready process
    while (!run_queue_.front().ready) {
        // if the first process is not ready, move it to the end of the queue and give its budget
        PID pid = run_queue_.front().pid;
        int64_t budget = run_queue_.front().budget_us;
        int64_t new_budget = GetBudgetUs(pid);
        // budget decay to avoid budget hoarding
        if (budget > new_budget) budget = new_budget + (budget - new_budget) * 0.9;
        budget += new_budget;

        run_queue_.pop_front();
        run_queue_.emplace_back(RunEntry{.pid=pid,.ready=false,.running=false,.budget_us=budget});
    }

    // run the first process
    PID pid = run_queue_.front().pid;
    int64_t budget = run_queue_.front().budget_us;
    int64_t new_budget = GetBudgetUs(pid);
    // budget decay to avoid budget hoarding
    if (budget > new_budget) budget = new_budget + (budget - new_budget) * 0.9;
    budget += new_budget;
    timeslice_end_ = now + microseconds(budget);
    AddTimer(timeslice_end_);
    SwitchProcess(pid, status);
}

void ProcessUtilizationPartitionPolicy::RecvHint(std::shared_ptr<const Hint> hint)
{
    switch (hint->Type())
    {
    case kHintTypeUtilization:
    {
        auto h = std::dynamic_pointer_cast<const UtilizationHint>(hint);
        XASSERT(h != nullptr, "hint type not match");
        PID pid = h->Pid();
        if (pid == 0) break;
        Utilization util = h->Util();
        if (util < UTILIZATION_MIN || util > UTILIZATION_MAX) {
            XWARN("invalid utilization %d", util);
            break;
        }
        utils_[pid] = util;
        XINFO("utilization of process %u set to %d", pid, util);
        break;
    }
    case kHintTypeTimeslice:
    {
        auto h = std::dynamic_pointer_cast<const TimesliceHint>(hint);
        XASSERT(h != nullptr, "hint type not match");
        Timeslice ts_us = h->Ts();
        if (ts_us < TIMESLICE_MIN || ts_us > TIMESLICE_MAX) {
            XWARN("invalid timeslice %ld", ts_us);
            break;
        }
        timeslice_avg_us_ = ts_us;
        XINFO("timeslice set to %ld us", ts_us);
        break;
    }
    default:
        XWARN("unsupported hint type: %d", hint->Type());
        break;
    }
}

int64_t ProcessUtilizationPartitionPolicy::GetBudgetUs(PID pid)
{
    int64_t total_util = 0;
    for (const auto &process : run_queue_) {
        auto it = utils_.find(process.pid);
        if (it == utils_.end()) {
            total_util += UTILIZATION_DEFAULT;
        } else {
            total_util += it->second;
        }
    }
    if (total_util == 0) return 0;

    int64_t pid_util = 0;
    auto it = utils_.find(pid);
    if (it == utils_.end()) pid_util = UTILIZATION_DEFAULT;
    else pid_util = it->second;

    return (int64_t)timeslice_avg_us_ * run_queue_.size() * pid_util / total_util;
}

bool ProcessUtilizationPartitionPolicy::ProcessReady(PID pid, const Status &status)
{
    const auto it = status.process_status.find(pid);
    XASSERT(it != status.process_status.end(), "process %u not found", pid);

    // if one xqueue is ready, the process is ready
    for (auto xq: it->second->running_xqueues) {
        const auto xq_it = status.xqueue_status.find(xq);
        if (xq_it == status.xqueue_status.end()) continue;
        if (xq_it->second->ready) return true;
    }
    for (auto xq: it->second->suspended_xqueues) {
        const auto xq_it = status.xqueue_status.find(xq);
        if (xq_it == status.xqueue_status.end()) continue;
        if (xq_it->second->ready) return true;
    }
    return false;
}

void ProcessUtilizationPartitionPolicy::SwitchProcess(PID pid, const Status &status)
{
    for (auto &process : run_queue_) {
        const auto it = status.process_status.find(process.pid);
        if (it == status.process_status.end()) continue;

        if (process.pid == pid) {
            // resume all suspended xqueues
            std::list<XQueueHandle> suspended_xqueues;
            for (const auto &xq: it->second->suspended_xqueues) {
                suspended_xqueues.push_back(xq);
            }
            for (const auto xq: suspended_xqueues) { this->Resume(xq); }
            process.running = true;
        } else {
            // suspend all running xqueues
            std::list<XQueueHandle> running_xqueues;
            for (const auto &xq: it->second->running_xqueues) {
                running_xqueues.push_back(xq);
            }
            for (const auto xq: running_xqueues) { this->Suspend(xq); }
            process.running = false;
        }
    }
}
