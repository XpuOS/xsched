#include <cmath>
#include <limits>
#include <map>
#include <set>

#include "xsched/utils/xassert.h"
#include "xsched/sched/policy/cfs.h"

using namespace xsched::sched;

void CompletelyFairSchedulerPolicy::Sched(const Status &status)
{
    auto now = std::chrono::system_clock::now();
    bool has_ready_tasks = false;

    for (auto &st : status.xqueue_status) {
        PID pid = st.second->pid;
        XQueueHandle handle = st.second->handle;

        auto pit = pending_hints_.find(handle);
        if (pit != pending_hints_.end()) {
            cfs_infos_[pid].priority = pit->second.first;
            cfs_infos_[pid].weight = pit->second.second;
            pending_hints_.erase(pit);
        }

        auto it = cfs_infos_.find(pid);
        if (it != cfs_infos_.end() && it->second.is_running) {
            auto delta_us = std::chrono::duration_cast<std::chrono::microseconds>(now - it->second.last_resume_time).count();
            it->second.vruntime += delta_us * (1024.0 / it->second.weight);
        }
    }

    std::map<XDevice, PID> min_vruntime_pids;
    std::map<XDevice, double> min_vruntimes;

    for (auto &st : status.xqueue_status) {
        if (!st.second->ready) continue;
        has_ready_tasks = true;

        XDevice device = st.second->device;
        PID pid = st.second->pid;

        auto it = cfs_infos_.find(pid);
        if (it != cfs_infos_.end()) {
            double current_vruntime = it->second.vruntime;
            if (min_vruntimes.find(device) == min_vruntimes.end() || current_vruntime < min_vruntimes[device]) {
                min_vruntimes[device] = current_vruntime;
            }
        }
    }

    for (auto &st : status.xqueue_status) {
        if (!st.second->ready) continue;

        XDevice device = st.second->device;
        PID pid = st.second->pid;

        if (cfs_infos_.find(pid) == cfs_infos_.end()) {
            cfs_infos_[pid] = CFSNode();
            cfs_infos_[pid].last_resume_time = now;
            if (min_vruntimes.find(device) != min_vruntimes.end()) {
                cfs_infos_[pid].vruntime = min_vruntimes[device];
            } else {
                cfs_infos_[pid].vruntime = 0.0;
            }
            if (min_vruntimes.find(device) == min_vruntimes.end()) {
                min_vruntimes[device] = cfs_infos_[pid].vruntime;
            }
        }

        double current_vruntime = cfs_infos_[pid].vruntime;

        if (min_vruntime_pids.find(device) == min_vruntime_pids.end() ||
            current_vruntime < cfs_infos_[min_vruntime_pids[device]].vruntime) {
            min_vruntime_pids[device] = pid;
        }
    }

    for (auto &st : status.xqueue_status) {
        PID pid = st.second->pid;
        if (!st.second->ready) {
            cfs_infos_[pid].is_running = false;
            continue;
        }
    }

    std::set<PID> running_pids;
    for (const auto &pair : min_vruntime_pids) {
        PID best_pid = pair.second;
        auto &node = cfs_infos_[best_pid];
        if (!node.is_running) {
            node.is_running = true;
            node.last_resume_time = now;
        } else {
            node.last_resume_time = now;
        }
        running_pids.insert(best_pid);
    }

    for (auto &st : status.xqueue_status) {
        PID pid = st.second->pid;
        if (!st.second->ready) continue;

        if (running_pids.count(pid)) {
            this->Resume(st.first);
        } else {
            this->Suspend(st.first);
            cfs_infos_[pid].is_running = false;
        }
    }

    if (has_ready_tasks) {
        this->AddTimer(now + time_slice_);
    }
}

void CompletelyFairSchedulerPolicy::RecvHint(std::shared_ptr<const Hint> hint)
{
    if (hint->Type() != kHintTypePriority) return;
    auto h = std::dynamic_pointer_cast<const PriorityHint>(hint);
    if (h == nullptr) return;

    Priority prio = h->Prio();
    double weight = 1024.0 * std::pow(1.2, prio);

    pending_hints_[h->Handle()] = {prio, weight};
    
    XINFO("CFS: set priority %d (weight %.2f) for XQueue 0x" FMT_64X, prio, weight, h->Handle());
}
