#include <cstdlib>

#include "xsched/utils/log.h"
#include "xsched/utils/xassert.h"
#include "xsched/sched/protocol/names.h"
#include "xsched/sched/policy/policy.h"
#include "xsched/sched/policy/hpf.h"
#include "xsched/sched/policy/up.h"
#include "xsched/sched/policy/pup.h"
#include "xsched/sched/policy/edf.h"
#include "xsched/sched/policy/lax.h"
#include "xsched/sched/policy/kedf.h"
// NEW_POLICY: New policy headers go here.

using namespace xsched::sched;

void Policy::Suspend(XQueueHandle xqueue)
{
    if (suspend_func_) return suspend_func_(xqueue);
    XDEBG("suspend function not set");
}

void Policy::Resume(XQueueHandle xqueue)
{
    if (resume_func_) return resume_func_(xqueue);
    XDEBG("resume function not set");
}

void Policy::AddTimer(const TimePoint time_point)
{
    if (add_timer_func_) return add_timer_func_(time_point);
    XDEBG("add timer function not set");
}

std::unique_ptr<Policy> xsched::sched::CreatePolicy(PolicyType type)
{
    // NEW_POLICY: A new case handling new PolicyType should be added here
    // when creating a new policy.
    switch (type) {
        case kPolicyTypeHighestPriorityFirst:
            return std::make_unique<HighestPriorityFirstPolicy>();
        case kPolicyTypeUtilizationPartition:
            return std::make_unique<UtilizationPartitionPolicy>();
        case kPolicyTypeProcessUtilizationPartition:
            return std::make_unique<ProcessUtilizationPartitionPolicy>();
        case kPolicyTypeEarlyDeadlineFirst:
            return std::make_unique<EarliestDeadlineFirstPolicy>();
        case kPolicyTypeLaxity:
            return std::make_unique<LaxityPolicy>();
        case kPolicyTypeKEarlyDeadlineFirst:
            return std::make_unique<KEarliestDeadlineFirstPolicy>();
        // NEW_POLICY: New PolicyTypes handling goes here.
        default:
            XASSERT(false, "invalid policy type: %d", type);
            return nullptr;
    }
}
