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
    char *scheduler_name = std::getenv(XSCHED_SCHEDULER_ENV_NAME);
    if (scheduler_name == nullptr) return std::make_shared<AppManagedScheduler>();

    SchedulerType scheduler_type = GetSchedulerType(scheduler_name);
    if (scheduler_type == kSchedulerTypeUnknown || scheduler_type == kSchedulerTypeAppManaged) {
        return std::make_shared<AppManagedScheduler>();
    }
    if (scheduler_type == kSchedulerTypeGlobal) return std::make_shared<GlobalScheduler>();

    char *policy_name = std::getenv(XSCHED_POLICY_ENV_NAME);
    if (policy_name == nullptr) return std::make_shared<AppManagedScheduler>();

    PolicyType policy_type = GetPolicyType(policy_name);
    if (policy_type == kPolicyTypeUnknown) {
        return std::make_shared<AppManagedScheduler>();
    }

    XASSERT(policy_type > kPolicyTypeUnknown && policy_type < kPolicyTypeMax, "must be a customized policy");
    return std::make_shared<LocalScheduler>(policy_type);
}
