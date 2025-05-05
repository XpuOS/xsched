#pragma once

#include <unordered_map>

#include "xsched/types.h"
#include "xsched/sched/policy/policy.h"
#include "xsched/sched/protocol/hint.h"

namespace xsched::sched
{

class HighestPriorityFirstPolicy : public Policy
{
public:
    HighestPriorityFirstPolicy();
    virtual ~HighestPriorityFirstPolicy() = default;

    virtual void Sched(const Status &status) override;
    virtual void RecvHint(std::shared_ptr<const Hint> hint) override;

private:
    enum Mode
    {
        kModeDefault, // each device schedules independently
        kModeCosched, // high priority task of XPU-a can preempt low priority task of XPU-b
    };

    Mode mode_;
    std::unordered_map<XQueueHandle, Priority> priorities_;
};

} // namespace xsched::sched
