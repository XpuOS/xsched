#pragma once

#include <map>
#include <string>

#include "xsched/types.h"
#include "xsched/sched/policy/policy.h"
#include "xsched/sched/scheduler/scheduler.h"

namespace xsched::sched
{

PolicyType GetPolicyType(const std::string &name);
const std::string &GetPolicyTypeName(PolicyType type);

SchedulerType GetSchedulerType(const std::string &name);
const std::string &GetSchedulerTypeName(SchedulerType type);

} // namespace xsched::sched
