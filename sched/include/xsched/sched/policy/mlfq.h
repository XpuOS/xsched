#pragma once

#include <map>
#include <chrono>

#include "xsched/sched/policy.h"

namespace xsched {
namespace sched {

struct MLFQNode {
    int priority = 0;               // Current priority level (0 is highest)
    bool is_running = false;        // Whether it's currently running
    bool was_ready_last_tick = false; // Edge detection for idle/ready

    using TimePoint = std::chrono::time_point<std::chrono::system_clock>;
    TimePoint i_a;                  // (1) Time became idle
    TimePoint p_a;                  // (2) Time of last priority update
    TimePoint q_a;                  // (3) Time of most recent request
    
    // For pending time tracking and time slice tracking
    std::chrono::microseconds accumulated_pending_time{0};
    TimePoint last_pending_start;   // When it entered pending state

    std::chrono::microseconds time_slice_used{0};
    TimePoint last_resume_time;     // When it started running
};

class MultiLevelFeedbackQueuePolicy : public Policy {
public:
    MultiLevelFeedbackQueuePolicy() = default;
    ~MultiLevelFeedbackQueuePolicy() override = default;

    void Sched(const Status &status) override;
    void RecvHint(std::shared_ptr<const Hint> hint) override;

private:
    std::map<XQueueHandle, MLFQNode> mlfq_infos_;
    
    // Configuration
    const int max_priority_ = 3;    // Levels 0, 1, 2, 3
    const std::chrono::microseconds recovery_threshold_{100000}; // 100ms
    const std::chrono::microseconds default_tick_{5000}; // 5ms scheduling tick

    // Get time slice based on priority level
    std::chrono::microseconds get_time_slice(int prio) const {
        // e.g., prio 0: 10ms, prio 1: 20ms, prio 2: 40ms, prio 3: 80ms
        return std::chrono::microseconds(10000 * (1 << prio));
    }
};

} // namespace sched
} // namespace xsched
