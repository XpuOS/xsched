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

    // 为正在运行的队列更新 vruntime
    for (auto &st : status.xqueue_status) {
        XQueueHandle handle = st.second->handle;
        auto it = cfs_infos_.find(handle);
        if (it != cfs_infos_.end() && it->second.is_running) {
            // 算出距离上次 Resume 过去了多少物理时间（ms）
            auto delta_us = std::chrono::duration_cast<std::chrono::microseconds>(now - it->second.last_resume_time).count();
            // 更新虚拟时间：真实时间 * (基准权重 / weight)
            it->second.vruntime += delta_us * (1024.0 / it->second.weight);
        }
    }

    // 找出每个 device 上，vruntime 最小的 ready task
    std::map<XDevice, XQueueHandle> min_vruntime_handles;
    std::map<XDevice, double> min_vruntimes;

    // 先遍历一遍已有的任务，找出当前每张显卡的 min_vruntime
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

    // 再遍历一遍，处理新任务的初始化，并选出最终应该运行的任务
    for (auto &st : status.xqueue_status) {
        if (!st.second->ready) continue;

        XDevice device = st.second->device;
        XQueueHandle handle = st.second->handle;

        // 如果是第一次见到这个队列，初始化它的 CFS 信息
        if (cfs_infos_.find(handle) == cfs_infos_.end()) {
            cfs_infos_[handle] = CFSNode();
            cfs_infos_[handle].last_resume_time = now;
            // 继承当前设备的最小 vruntime，防止饿死老任务
            if (min_vruntimes.find(device) != min_vruntimes.end()) {
                cfs_infos_[handle].vruntime = min_vruntimes[device];
            } else {
                cfs_infos_[handle].vruntime = 0.0;
            }
            // 确保 min_vruntimes 包含该新任务的 vruntime（主要针对 device 上没有老任务的情况）
            if (min_vruntimes.find(device) == min_vruntimes.end()) {
                min_vruntimes[device] = cfs_infos_[handle].vruntime;
            }
        }

        double current_vruntime = cfs_infos_[handle].vruntime;

        // 找这块显卡上最终的最小值和对应的 handle
        if (min_vruntime_handles.find(device) == min_vruntime_handles.end() || current_vruntime < cfs_infos_[min_vruntime_handles[device]].vruntime) {
            min_vruntime_handles[device] = handle;
        }
    }

    // Resume 那个最小 vruntime 的任务，Suspend 其他所有任务
    for (auto &st : status.xqueue_status) {
        XDevice device = st.second->device;
        XQueueHandle handle = st.second->handle;

        if (!st.second->ready) {
            cfs_infos_[handle].is_running = false;
            continue;
        }

        if (min_vruntime_handles[device] == handle) {
            // 被选中的 vruntime 最小的 task
            if (!cfs_infos_[handle].is_running) {
                this->Resume(handle);
                cfs_infos_[handle].is_running = true;
                cfs_infos_[handle].last_resume_time = now; // 重新计时
            } else {
                // 更新时间以防重复累加
                cfs_infos_[handle].last_resume_time = now;
            }
        } else {
            // 其他 task 必须停止运行
            this->Suspend(handle);
            cfs_infos_[handle].is_running = false;
        }
    }

    // 强制 5ms 后重新进行一轮调度
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

    // Weight：这里 base 为 1024。通过 prio 计算 weight，prio 每增加 1，权重增加 20%
    // 权重越大，vruntime 涨得越慢，从而能分到更多物理运行时间
    double weight = 1024.0 * std::pow(1.2, prio);

    cfs_infos_[handle].priority = prio;
    cfs_infos_[handle].weight = weight;
    
    XINFO("CFS: set priority %d (weight %.2f) for XQueue 0x" FMT_64X, prio, weight, handle);
}
