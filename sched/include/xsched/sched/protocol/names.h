#pragma once

#include <map>
#include <string>

#include "xsched/types.h"
#include "xsched/sched/policy/policy.h"
#include "xsched/sched/scheduler/scheduler.h"

namespace xsched::sched
{

SchedulerType GetSchedulerType(const std::string &name);
const std::string &GetSchedulerTypeName(SchedulerType type);

PolicyType GetPolicyType(const std::string &name);
const std::string &GetPolicyTypeName(PolicyType type);

} // namespace xsched::sched
