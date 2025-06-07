#include <cstdlib>

#include "xsched/utils/log.h"
#include "xsched/protocol/names.h"
#include "xsched/sched/scheduler/local.h"
#include "xsched/sched/scheduler/global.h"
#include "xsched/sched/scheduler/scheduler.h"
#include "xsched/sched/scheduler/app_managed.h"

using namespace xsched::sched;
using namespace xsched::protocol;

void Scheduler::Execute(std::shared_ptr<const Operation> operation)
{
    if (executor_) return executor_(operation);
    XDEBG("executor not set");
}

std::shared_ptr<Scheduler>
xsched::sched::CreateScheduler(XSchedulerType scheduler_type, XPolicyType policy_type)
{
    if (policy_type > kPolicyUnknown && policy_type < kPolicyMax) {
        XINFO("using local scheduler with policy %s", GetPolicyTypeName(policy_type).c_str());
        return std::make_shared<LocalScheduler>(policy_type);
    }

    if (scheduler_type == kSchedulerGlobal) {
        XINFO("using global scheduler");
        return std::make_shared<GlobalScheduler>();
    }

    XINFO("using app-managed scheduler");
    return std::make_shared<AppManagedScheduler>();
}
