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
    char *policy_name = std::getenv(XSCHED_POLICY_ENV_NAME);
    if (policy_name == nullptr) return std::make_shared<AppManagedScheduler>();

    PolicyType type = GetPolicyType(policy_name);
    if (type == kPolicyTypeUnknown || type == kPolicyTypeAppManaged) {
        return std::make_shared<AppManagedScheduler>();
    }

    if (type == kPolicyTypeGlobal) return std::make_shared<GlobalScheduler>();

    XASSERT(type > kPolicyTypeInternalMax && type < kPolicyTypeMax, "must be a customized policy");
    return std::make_shared<LocalScheduler>(type);
}
