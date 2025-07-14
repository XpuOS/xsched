#pragma once

#include "xsched/types.h"
#include "xsched/sched/policy/policy.h"
#include "xsched/sched/protocol/hint.h"

namespace xsched::sched
{

class CPUHighestPriorityFirstPolicy : public Policy
{
public:
    CPUHighestPriorityFirstPolicy(): Policy(kPolicyCPUHighestPriorityFirst) {}
    virtual ~CPUHighestPriorityFirstPolicy() = default;

    virtual void Sched(const Status &status) override;
    virtual void RecvHint(std::shared_ptr<const Hint> hint) override;

private:
};

} // namespace xsched::sched
