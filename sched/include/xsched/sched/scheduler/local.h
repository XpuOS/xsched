#pragma once

#include <list>
#include <queue>
#include <mutex>
#include <thread>
#include <chrono>
#include <memory>
#include <condition_variable>

#include "xsched/utils/common.h"
#include "xsched/sched/policy/policy.h"
#include "xsched/sched/protocol/status.h"
#include "xsched/sched/scheduler/scheduler.h"

namespace xsched::sched
{

class LocalScheduler : public Scheduler
{
public:
    LocalScheduler(PolicyType type);
    virtual ~LocalScheduler();

    virtual void Run() override;
    virtual void Stop() override;
    virtual void RecvEvent(std::shared_ptr<const Event> event) override;

    void SetPolicy(PolicyType type);
    PolicyType GetPolicy() const { return policy_type_; }

private:
    void Worker();
    void ExecuteOperations();
    void CreateXQueueStatus(PID pid, const std::string &cmdline, XQueueHandle handle, XDevice dev,
                            XPreemptLevel level, int64_t threshold, int64_t batch_size,
                            bool ready, std::chrono::system_clock::time_point ready_time);
    void UpdateStatus(std::shared_ptr<const Event> event);
    void Suspend(XQueueHandle handle);
    void Resume(XQueueHandle handle);
    void AddTimer(const std::chrono::system_clock::time_point time_point);

    PolicyType policy_type_;
    std::unique_ptr<Policy> policy_ = nullptr;
    std::unique_ptr<std::thread> thread_ = nullptr;

    Status status_;
    std::mutex event_mtx_;
    std::condition_variable event_cv_;
    std::unique_ptr<std::list<std::shared_ptr<const Event>>> event_queue_ = nullptr;
    std::deque<std::chrono::system_clock::time_point> timers_;
};

} // namespace xsched::sched
