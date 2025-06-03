#pragma once

#include <unordered_map>

#include "xsched/types.h"
#include "xsched/sched/policy/policy.h"
#include "xsched/sched/protocol/hint.h"

namespace xsched::sched
{

// K-Earliest Deadline First
// Always schedule K tasks with the earliest deadline first.
class KEarliestDeadlineFirstPolicy : public Policy
{
public:
    KEarliestDeadlineFirstPolicy();
    KEarliestDeadlineFirstPolicy(int K): Policy(kPolicyTypeKEarlyDeadlineFirst), K_(K) {}
    virtual ~KEarliestDeadlineFirstPolicy() = default;

    virtual void Sched(const Status &status) override;
    virtual void RecvHint(std::shared_ptr<const Hint> hint) override;

private:
    std::unordered_map<XQueueHandle, Deadline> deadlines_;
    int K_;
};

} // namespace xsched::sched
