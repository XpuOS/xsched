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

    // 为运行的队列更新 vruntime
    for (auto &st : status.xqueue_status) {
        XQueueHandle handle = st.second->handle;
        auto it = cfs_infos_.find(handle);
        if (it != cfs_infos_.end() && it->second.is_running) {
            // 算出距离上次 Resume 过去了多少物理时间
            auto delta_us = std::chrono::duration_cast<std::chrono::microseconds>(now - it->second.last_resume_time).count();
            // 更新 vruntime：真实时间 * (基准权重 / 当前任务权重)
            it->second.vruntime += delta_us * (1024.0 / it->second.weight);
        }
    }

    // 找出每个 device 上，vruntime 最小的就绪任务
    std::map<XDevice, XQueueHandle> min_vruntime_handles;
    std::map<XDevice, double> min_vruntimes;

    for (auto &st : status.xqueue_status) {
        if (!st.second->ready) continue;
        has_ready_tasks = true;

        XDevice device = st.second->device;
        XQueueHandle handle = st.second->handle;

        // 如果是第一次见到这个队列，初始化它的 CFS 信息
        if (cfs_infos_.find(handle) == cfs_infos_.end()) {
            cfs_infos_[handle] = CFSNode();
            cfs_infos_[handle].last_resume_time = now;
            // 这里为了简单，新来的任务的 vruntime 就直接从 0 开始。在真实 Linux CFS 中，它会等于当前的 min_vruntime。
            // TODO：更新这里vruntime设置思路，将当前任务的 vruntime 设置为该进程最小的 vruntime！
        }

        double current_vruntime = cfs_infos_[handle].vruntime;

        // 找出 device 上的运行时间最小值和对应任务 handle
        if (min_vruntimes.find(device) == min_vruntimes.end() || current_vruntime < min_vruntimes[device]) {
            min_vruntimes[device] = current_vruntime;
            min_vruntime_handles[device] = handle;
        }
    }

    // 抢占：Resume 那个最小 vruntime 的任务，Suspend 其他所有任务
    for (auto &st : status.xqueue_status) {
        XDevice device = st.second->device;
        XQueueHandle handle = st.second->handle;

        if (!st.second->ready) {
            cfs_infos_[handle].is_running = false;
            continue;
        }

        if (min_vruntime_handles[device] == handle) {
            // 拿出当前 device 上 vruntime 最短的 task
            if (!cfs_infos_[handle].is_running) {
                this->Resume(handle);
                cfs_infos_[handle].is_running = true;
                cfs_infos_[handle].last_resume_time = now; // 对该任务重新计时
            } else {
                // 即使该任务上一轮已经在运行，还需要更新resume time防止重复计算虚拟运行时间
                cfs_infos_[handle].last_resume_time = now;
            }
        } else {
            // 挂起其他所有进程
            this->Suspend(handle);
            cfs_infos_[handle].is_running = false;
        }
    }

    // 定时器中断：强制 1ms 重新进行一轮调度
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

    // 计算权重：这里假设基准为 1024。优先级数字每大 1，权重增加 20%
    // 权重越大，vruntime 涨得越慢，从而可以保证对应的任务可以分配到更多物理运行时间
    double weight = 1024.0 * std::pow(1.2, prio);

    cfs_infos_[handle].priority = prio;
    cfs_infos_[handle].weight = weight;
    
    XINFO("CFS: set priority %d (weight %.2f) for XQueue 0x" FMT_64X, prio, weight, handle);
}
