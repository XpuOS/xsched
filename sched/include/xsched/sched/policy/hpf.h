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
    HighestPriorityFirstPolicy(): Policy(kPolicyHighestPriorityFirst) {}
    virtual ~HighestPriorityFirstPolicy() = default;

    virtual void Sched(const Status &status) override;
    virtual void RecvHint(std::shared_ptr<const Hint> hint) override;

private:
    Priority GetPriority(XQueueHandle handle);
    std::unordered_map<XQueueHandle, Priority> priorities_;
};

} // namespace xsched::sched
