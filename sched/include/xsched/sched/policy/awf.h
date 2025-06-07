#pragma once

#include <unordered_set>
#include <unordered_map>

#include "xsched/types.h"
#include "xsched/sched/policy/policy.h"
#include "xsched/sched/protocol/hint.h"

namespace xsched::sched
{

class ActiveWindowFirstPolicy : public Policy
{
public:
    ActiveWindowFirstPolicy(): Policy(kPolicyActiveWindowFirst) {}
    virtual ~ActiveWindowFirstPolicy() = default;

    virtual void Sched(const Status &status) override;
    virtual void RecvHint(std::shared_ptr<const Hint> hint) override;

private:
    std::unordered_set<PID> active_pids_;
    std::unordered_map<uint64_t, PID> display_active_pids_; // display id => active pid
};

} // namespace xsched::sched
