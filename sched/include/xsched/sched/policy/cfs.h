#pragma once

#include <unordered_map>
#include <chrono>

#include "xsched/types.h"
#include "xsched/sched/policy/policy.h"
#include "xsched/sched/protocol/hint.h"

namespace xsched::sched {

struct CFSNode {
    double weight = 1024.0;
    double vruntime = 0.0;
    std::chrono::system_clock::time_point last_resume_time;
    bool is_running = false;
    Priority priority = PRIORITY_DEFAULT;
};

class CompletelyFairSchedulerPolicy : public Policy {
public:
    CompletelyFairSchedulerPolicy(): Policy(kPolicyCompletelyFairScheduler) {} 
    virtual ~CompletelyFairSchedulerPolicy() = default;

    virtual void Sched(const Status &status) override;
    virtual void RecvHint(std::shared_ptr<const Hint> hint) override;

private:
    std::unordered_map<XQueueHandle, CFSNode> cfs_infos_;
    std::chrono::microseconds time_slice_{1000}; // 设置 1ms 的时间片
};

} // namespace xsched::sched
