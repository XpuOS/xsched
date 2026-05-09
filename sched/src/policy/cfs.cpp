#include <cmath>
#include <limits>
#include <map>

#include "xsched/utils/xassert.h"
#include "xsched/sched/policy/cfs.h"

using namespace xsched::sched;

void CompletelyFairSchedulerPolicy::Sched(const Status &status)
{
    auto now = std::chrono::system_clock::now();
    bool has_ready_tasks = false;

    // update vruntime for running queues
    for (auto &st : status.xqueue_status) {
        XQueueHandle handle = st.second->handle;
        auto it = cfs_infos_.find(handle);
        if (it != cfs_infos_.end() && it->second.is_running) {
            auto delta_us = std::chrono::duration_cast<std::chrono::microseconds>(now - it->second.last_resume_time).count();
            // Update virtual time: real time * (base weight / current task weight)
            it->second.vruntime += delta_us * (1024.0 / it->second.weight);
        }
    }

    // Find the ready task with the minimum vruntime on each physical GPU
    std::map<XDevice, XQueueHandle> min_vruntime_handles;
    std::map<XDevice, double> min_vruntimes;

    // find the current min_vruntime for each device among existing tasks
    for (auto &st : status.xqueue_status) {
        if (!st.second->ready) continue;
        has_ready_tasks = true;

        XDevice device = st.second->device;
        XQueueHandle handle = st.second->handle;

        auto it = cfs_infos_.find(handle);
        if (it != cfs_infos_.end()) {
            double current_vruntime = it->second.vruntime;
            if (min_vruntimes.find(device) == min_vruntimes.end() || current_vruntime < min_vruntimes[device]) {
                min_vruntimes[device] = current_vruntime;
            }
        }
    }

    // handle initialization for new tasks and select the final task to run
    for (auto &st : status.xqueue_status) {
        if (!st.second->ready) continue;

        XDevice device = st.second->device;
        XQueueHandle handle = st.second->handle;

        // If this is the first time seeing this queue, initialize its CFS info
        if (cfs_infos_.find(handle) == cfs_infos_.end()) {
            cfs_infos_[handle] = CFSNode();
            cfs_infos_[handle].last_resume_time = now;
            // Inherit the minimum vruntime of the current device to prevent new tasks from starving old tasks
            if (min_vruntimes.find(device) != min_vruntimes.end()) {
                cfs_infos_[handle].vruntime = min_vruntimes[device];
            } else {
                cfs_infos_[handle].vruntime = 0.0;
            }
            // Ensure min_vruntimes contains the vruntime of the new task (mainly for cases where the device has no old tasks)
            if (min_vruntimes.find(device) == min_vruntimes.end()) {
                min_vruntimes[device] = cfs_infos_[handle].vruntime;
            }
        }

        double current_vruntime = cfs_infos_[handle].vruntime;

        // Find the final minimum value and corresponding handle on this device
        if (min_vruntime_handles.find(device) == min_vruntime_handles.end() || current_vruntime < cfs_infos_[min_vruntime_handles[device]].vruntime) {
            min_vruntime_handles[device] = handle;
        }
    }

    // Resume the task with the minimum vruntime, Suspend all others
    for (auto &st : status.xqueue_status) {
        XDevice device = st.second->device;
        XQueueHandle handle = st.second->handle;

        if (!st.second->ready) {
            cfs_infos_[handle].is_running = false;
            continue;
        }

        if (min_vruntime_handles[device] == handle) {
            if (!cfs_infos_[handle].is_running) {
                this->Resume(handle);
                cfs_infos_[handle].is_running = true;
                cfs_infos_[handle].last_resume_time = now;
            } else {
                cfs_infos_[handle].last_resume_time = now;
            }
        } else {
            this->Suspend(handle);
            cfs_infos_[handle].is_running = false;
        }
    }

    // force a new scheduling round after the time slice
    if (has_ready_tasks) {
        this->AddTimer(now + time_slice_);
    }
}

void CompletelyFairSchedulerPolicy::RecvHint(std::shared_ptr<const Hint> hint)
{
    if (hint->Type() != kHintTypePriority) return;
    auto h = std::dynamic_pointer_cast<const PriorityHint>(hint);
    if (h == nullptr) return;

    XQueueHandle handle = h->Handle();
    Priority prio = h->Prio();

    // Calculate weight: assuming base is 1024. For each priority increase, weight increases by 20%
    double weight = 1024.0 * std::pow(1.2, prio);

    cfs_infos_[handle].priority = prio;
    cfs_infos_[handle].weight = weight;
    
    XINFO("CFS: set priority %d (weight %.2f) for XQueue 0x" FMT_64X, prio, weight, handle);
}
