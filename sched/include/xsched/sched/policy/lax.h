#pragma once

#include <unordered_map>

#include "xsched/types.h"
#include "xsched/sched/policy/policy.h"
#include "xsched/sched/protocol/hint.h"

namespace xsched::sched
{

class LaxityPolicy : public Policy
{
public:
    LaxityPolicy(): Policy(kPolicyTypeLaxity) {}
    virtual ~LaxityPolicy() = default;

    virtual void Sched(const Status &status) override;
    virtual void RecvHint(std::shared_ptr<const Hint> hint) override;

private:
    struct LaxityInfo
    {
        Laxity   lax;
        Priority lax_prio;
        Priority crit_prio;
    };
    std::unordered_map<XQueueHandle, LaxityInfo> laxity_infos_;
};

} // namespace xsched::sched
