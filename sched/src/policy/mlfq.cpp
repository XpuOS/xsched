#include "xsched/sched/policy/mlfq.h"
#include "xsched/utils/log.h"
#include <map>

using namespace xsched::sched;

void MultiLevelFeedbackQueuePolicy::Sched(const Status &status)
{
    auto now = std::chrono::system_clock::now();
    bool has_ready_tasks = false;

    // Maintain variables: i_a (idle time) and q_a (request time) through edge detection
    for (auto &st : status.xqueue_status) {
        XQueueHandle handle = st.second->handle;
        bool is_ready_now = st.second->ready;
        
        // Initialize if not present
        if (mlfq_infos_.find(handle) == mlfq_infos_.end()) {
            mlfq_infos_[handle] = MLFQNode();
            mlfq_infos_[handle].p_a = now;
            mlfq_infos_[handle].i_a = now;
            mlfq_infos_[handle].q_a = now;
            mlfq_infos_[handle].last_pending_start = now;
            mlfq_infos_[handle].last_resume_time = now;
            XINFO("MLFQ: Initialized new XQueue 0x" FMT_64X, handle);
        }

        auto &node = mlfq_infos_[handle];

        if (is_ready_now && !node.was_ready_last_tick) {
            // Task just woke up (idle -> ready)
            node.q_a = now;
            node.last_pending_start = now;
        } 
        else if (!is_ready_now && node.was_ready_last_tick) {
            // Task just finished executing (ready -> idle)
            node.i_a = now;
        }
        node.was_ready_last_tick = is_ready_now;
    }

    // Count number of tasks (N) per priority level on each device
    std::map<XDevice, std::map<int, int>> N_count;
    for (auto &st : status.xqueue_status) {
        if (st.second->ready) {
            int prio = mlfq_infos_[st.second->handle].priority;
            N_count[st.second->device][prio]++;
        }
    }

    // Update pending times and check for priority recovery / demotion
    for (auto &st : status.xqueue_status) {
        XQueueHandle handle = st.second->handle;
        XDevice device = st.second->device;
        auto &node = mlfq_infos_[handle];

        if (!st.second->ready) continue;
        has_ready_tasks = true;

        // Pending time & Run time updates
        if (!node.is_running) {
            auto pending_duration = std::chrono::duration_cast<std::chrono::microseconds>(now - node.last_pending_start);
            node.accumulated_pending_time += pending_duration;
            node.last_pending_start = now; // reset start point
        } else {
            auto run_duration = std::chrono::duration_cast<std::chrono::microseconds>(now - node.last_resume_time);
            node.time_slice_used += run_duration;
            node.last_resume_time = now;
        }

        // a) Soft Priority Recovery
        int N = N_count[device][node.priority];
        double R = (N > 1) ? (0.9 / N) : 1.0; // Dynamic discount factor R < 1/N
        
        auto time_since_last_update = std::chrono::duration_cast<std::chrono::microseconds>(now - node.p_a).count();
        double discounted_pending = node.accumulated_pending_time.count() * R;
        double effective_time = time_since_last_update - discounted_pending;

        if (effective_time > recovery_threshold_.count()) {
            if (node.priority > 0) {
                node.priority--;
                XINFO("MLFQ: Priority Recovery (Promotion) for 0x" FMT_64X " to %d", handle, node.priority);
            }
            // Record time of last priority update
            node.p_a = now;
            node.accumulated_pending_time = std::chrono::microseconds(0);
            node.time_slice_used = std::chrono::microseconds(0);
        }

        // b) Time Slice Exhaustion (Demotion)
        if (node.is_running && node.time_slice_used >= get_time_slice(node.priority)) {
            if (node.priority < max_priority_) {
                node.priority++;
                XINFO("MLFQ: Time Slice Exhausted (Demotion) for 0x" FMT_64X " to %d", handle, node.priority);
            }
            node.p_a = now;
            node.time_slice_used = std::chrono::microseconds(0);
            node.accumulated_pending_time = std::chrono::microseconds(0);
            
            // Force it to yield
            node.is_running = false; 
            this->Suspend(handle);
            node.last_pending_start = now; // Starts pending from here
        }
    }

    // Select the highest priority task per device to execute
    std::map<XDevice, XQueueHandle> best_handles;
    std::map<XDevice, int> best_prios;

    for (auto &st : status.xqueue_status) {
        if (!st.second->ready) continue;
        
        XDevice device = st.second->device;
        XQueueHandle handle = st.second->handle;
        auto &node = mlfq_infos_[handle];
        
        // Find highest priority (lowest number)
        if (best_prios.find(device) == best_prios.end() || node.priority < best_prios[device]) {
            best_prios[device] = node.priority;
            best_handles[device] = handle;
        } 
        // Tie breaker: longest pending time (acts as Round-Robin or FCFS within priority)
        else if (node.priority == best_prios[device]) {
            if (node.accumulated_pending_time > mlfq_infos_[best_handles[device]].accumulated_pending_time) {
                best_handles[device] = handle;
            }
        }
    }

    // Suspend / Resume execution based on the selection
    for (auto &st : status.xqueue_status) {
        XDevice device = st.second->device;
        XQueueHandle handle = st.second->handle;
        auto &node = mlfq_infos_[handle];

        if (!st.second->ready) {
            node.is_running = false;
            continue;
        }

        if (best_handles[device] == handle) {
            // The chosen one
            if (!node.is_running) {
                this->Resume(handle);
                node.is_running = true;
                node.last_resume_time = now;
                node.last_pending_start = now; // no longer pending
            } else {
                // already running
                node.last_resume_time = now;
            }
        } else {
            // Not chosen, must wait
            if (node.is_running) {
                this->Suspend(handle);
                node.is_running = false;
                node.last_pending_start = now;
            }
        }
    }

    // Schedule next timer interrupt to keep evaluating time slices
    if (has_ready_tasks) {
        this->AddTimer(now + default_tick_);
    }
}

void MultiLevelFeedbackQueuePolicy::RecvHint(std::shared_ptr<const Hint> hint)
{
    (void)hint;
    // Not strictly needed for autonomous Soft Priority Recovery,
    // but can be implemented to support manual user priorities if required.
}