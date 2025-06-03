#pragma once

#include <map>
#include <string>

#include "xsched/types.h"
#include "xsched/sched/policy/policy.h"

namespace xsched::sched
{

PolicyType GetPolicyType(const std::string &name);
const std::string &GetPolicyTypeName(PolicyType type);

} // namespace xsched::sched
