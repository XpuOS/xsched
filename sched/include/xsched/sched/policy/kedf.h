#pragma once

#include <unordered_map>

#include "xsched/types.h"
#include "xsched/sched/policy/policy.h"
#include "xsched/sched/protocol/hint.h"

namespace xsched::sched
{

struct DeadlineEntry
{
    XQueueHandle xqueue;
    std::chrono::system_clock::time_point deadline;
};

class KEarliestDeadlineFirstPolicy : public Policy
{
public:
    KEarliestDeadlineFirstPolicy(): Policy(kPolicyKEarliestDeadlineFirst) {}
    virtual ~KEarliestDeadlineFirstPolicy() = default;

    virtual void Sched(const Status &status) override;
    virtual void RecvHint(std::shared_ptr<const Hint> hint) override;

private:
    size_t k_ = 1;
    std::unordered_map<XQueueHandle, Deadline> deadlines_;
};

} // namespace xsched::sched
