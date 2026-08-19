#pragma once

#include <map>
#include <chrono>

#include "xsched/sched/policy/policy.h"

namespace xsched {
namespace sched {

struct MLFQNode {
    int priority = 0;
    bool is_running = false;
    bool was_ready_last_tick = false;

    using TimePoint = std::chrono::time_point<std::chrono::system_clock>;
    TimePoint i_a;
    TimePoint p_a;
    TimePoint q_a;
    
    std::chrono::microseconds accumulated_pending_time{0};
    TimePoint last_pending_start;

    std::chrono::microseconds time_slice_used{0};
    TimePoint last_resume_time;
};

class MultiLevelFeedbackQueuePolicy : public Policy {
public:
    MultiLevelFeedbackQueuePolicy() : Policy(kPolicyMultiLevelFeedbackQueue) {}
    ~MultiLevelFeedbackQueuePolicy() override = default;

    void Sched(const Status &status) override;
    void RecvHint(std::shared_ptr<const Hint> hint) override;

private:
    std::map<PID, MLFQNode> mlfq_infos_;
    
    const int max_priority_ = 3;
    const std::chrono::microseconds recovery_threshold_{100000};
    const std::chrono::microseconds default_tick_{5000};

    std::chrono::microseconds get_time_slice(int prio) const {
        return std::chrono::microseconds(10000 * (1 << prio));
    }
};

} // namespace sched
} // namespace xsched
