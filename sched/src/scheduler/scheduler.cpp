#include <cstdlib>

#include "xsched/utils/log.h"
#include "xsched/utils/xassert.h"
#include "xsched/protocol/def.h"
#include "xsched/sched/protocol/names.h"
#include "xsched/sched/scheduler/local.h"
#include "xsched/sched/scheduler/global.h"
#include "xsched/sched/scheduler/scheduler.h"
#include "xsched/sched/scheduler/app_managed.h"

using namespace xsched::sched;

void Scheduler::Execute(std::shared_ptr<const Operation> operation)
{
    if (executor_) return executor_(operation);
    XDEBG("executor not set");
}

std::shared_ptr<Scheduler> xsched::sched::CreateScheduler()
{
    SchedulerType scheduler_type = kSchedulerUnknown;
    PolicyType policy_type = kPolicyUnknown;
    char *scheduler_str = std::getenv(XSCHED_SCHEDULER_ENV_NAME);
    char *policy_str = std::getenv(XSCHED_POLICY_ENV_NAME);
    if (scheduler_str) scheduler_type = GetSchedulerType(scheduler_str);
    if (policy_str) policy_type = GetPolicyType(policy_str);

    if (policy_type > kPolicyUnknown && policy_type < kPolicyMax) {
        XINFO("using local scheduler with policy %s", policy_str);
        return std::make_shared<LocalScheduler>(policy_type);
    }

    if (scheduler_type == kSchedulerGlobal) {
        XINFO("using global scheduler");
        return std::make_shared<GlobalScheduler>();
    }

    XINFO("using app-managed scheduler");
    return std::make_shared<AppManagedScheduler>();
}
