#include "xsched/sched/policy/mlfq.h"
#include "xsched/utils/log.h"
#include <map>
#include <set>

using namespace xsched::sched;

void MultiLevelFeedbackQueuePolicy::Sched(const Status &status)
{
    auto now = std::chrono::system_clock::now();
    bool has_ready_tasks = false;

    for (auto &st : status.xqueue_status) {
        PID pid = st.second->pid;
        bool is_ready_now = st.second->ready;
        
        if (mlfq_infos_.find(pid) == mlfq_infos_.end()) {
            mlfq_infos_[pid] = MLFQNode();
            mlfq_infos_[pid].p_a = now;
            mlfq_infos_[pid].i_a = now;
            mlfq_infos_[pid].q_a = now;
            mlfq_infos_[pid].last_pending_start = now;
            mlfq_infos_[pid].last_resume_time = now;
            XINFO("MLFQ: Initialized new PID " FMT_PID, pid);
        }

        auto &node = mlfq_infos_[pid];

        if (is_ready_now && !node.was_ready_last_tick) {
            node.q_a = now;
            node.last_pending_start = now;
        } 
        else if (!is_ready_now && node.was_ready_last_tick) {
            node.i_a = now;
        }
        node.was_ready_last_tick = is_ready_now;
    }

    std::map<XDevice, std::map<int, int>> N_count;
    for (auto &st : status.xqueue_status) {
        if (st.second->ready) {
            int prio = mlfq_infos_[st.second->pid].priority;
            N_count[st.second->device][prio]++;
        }
    }

    for (auto &st : status.xqueue_status) {
        PID pid = st.second->pid;
        XDevice device = st.second->device;
        auto &node = mlfq_infos_[pid];

        if (!st.second->ready) continue;
        has_ready_tasks = true;

        if (!node.is_running) {
            auto pending_duration = std::chrono::duration_cast<std::chrono::microseconds>(now - node.last_pending_start);
            node.accumulated_pending_time += pending_duration;
            node.last_pending_start = now;
        } else {
            auto run_duration = std::chrono::duration_cast<std::chrono::microseconds>(now - node.last_resume_time);
            node.time_slice_used += run_duration;
            node.last_resume_time = now;
        }

        int N = N_count[device][node.priority];
        double R = (N > 1) ? (0.9 / N) : 1.0;
        
        auto time_since_last_update = std::chrono::duration_cast<std::chrono::microseconds>(now - node.p_a).count();
        double discounted_pending = node.accumulated_pending_time.count() * R;
        double effective_time = time_since_last_update - discounted_pending;

        if (effective_time > recovery_threshold_.count()) {
            if (node.priority > 0) {
                node.priority--;
                XINFO("MLFQ: Priority Recovery (Promotion) for PID " FMT_PID " to %d", pid, node.priority);
            }
            node.p_a = now;
            node.accumulated_pending_time = std::chrono::microseconds(0);
            node.time_slice_used = std::chrono::microseconds(0);
        }

        if (node.is_running && node.time_slice_used >= get_time_slice(node.priority)) {
            if (node.priority < max_priority_) {
                node.priority++;
                XINFO("MLFQ: Time Slice Exhausted (Demotion) for PID " FMT_PID " to %d", pid, node.priority);
            }
            node.p_a = now;
            node.time_slice_used = std::chrono::microseconds(0);
            node.accumulated_pending_time = std::chrono::microseconds(0);
            
            node.is_running = false; 
            node.last_pending_start = now;
        }
    }

    std::map<XDevice, PID> best_pids;
    std::map<XDevice, int> best_prios;

    for (auto &st : status.xqueue_status) {
        if (!st.second->ready) continue;
        
        XDevice device = st.second->device;
        PID pid = st.second->pid;
        auto &node = mlfq_infos_[pid];
        
        if (best_prios.find(device) == best_prios.end() || node.priority < best_prios[device]) {
            best_prios[device] = node.priority;
            best_pids[device] = pid;
        } 
        else if (node.priority == best_prios[device]) {
            if (node.accumulated_pending_time > mlfq_infos_[best_pids[device]].accumulated_pending_time) {
                best_pids[device] = pid;
            }
        }
    }

    std::set<PID> running_pids;
    for (const auto &pair : best_pids) {
        PID best_pid = pair.second;
        auto &node = mlfq_infos_[best_pid];
        if (!node.is_running) {
            node.is_running = true;
            node.last_resume_time = now;
            node.last_pending_start = now;
        } else {
            node.last_resume_time = now;
        }
        running_pids.insert(best_pid);
    }

    for (auto &st : status.xqueue_status) {
        PID pid = st.second->pid;
        auto &node = mlfq_infos_[pid];

        if (!st.second->ready) {
            node.is_running = false;
            continue;
        }

        if (!running_pids.count(pid)) {
            if (node.is_running) {
                node.is_running = false;
                node.last_pending_start = now;
            }
        }
    }

    for (auto &st : status.xqueue_status) {
        PID pid = st.second->pid;
        if (!st.second->ready) continue;

        if (running_pids.count(pid)) {
            this->Resume(st.first);
        } else {
            this->Suspend(st.first);
        }
    }

    if (has_ready_tasks) {
        this->AddTimer(now + default_tick_);
    }
}

void MultiLevelFeedbackQueuePolicy::RecvHint(std::shared_ptr<const Hint> hint)
{
    (void)hint;
}
