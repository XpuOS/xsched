#pragma once

#include <unordered_map>

#include "xsched/types.h"
#include "xsched/sched/policy/policy.h"
#include "xsched/sched/protocol/hint.h"

namespace xsched::sched
{

class EarliestDeadlineFirstPolicy : public Policy
{
public:
    EarliestDeadlineFirstPolicy(): Policy(kPolicyTypeEarlyDeadlineFirst) {}
    virtual ~EarliestDeadlineFirstPolicy() = default;

    virtual void Sched(const Status &status) override;
    virtual void RecvHint(std::shared_ptr<const Hint> hint) override;

private:
    std::unordered_map<XQueueHandle, Deadline> deadlines_;
};

} // namespace xsched::sched
