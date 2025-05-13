#include <map>
#include "xsched/protocol/def.h"
#include "xsched/sched/protocol/names.h"

namespace xsched::sched
{

const std::map<PolicyType, std::string> &PolicyTypeNames() {
    static const std::map<PolicyType, std::string> kPolicyTypeNames {
        { kPolicyTypeUnknown                    , "Unknown"              },
        { kPolicyTypeHighestPriorityFirst       , XSCHED_POLICY_NAME_HPF },
        { kPolicyTypeUtilizationPartition       , XSCHED_POLICY_NAME_UP  },
        { kPolicyTypeProcessUtilizationPartition, XSCHED_POLICY_NAME_PUP },
        { kPolicyTypeEarlyDeadlineFirst         , XSCHED_POLICY_NAME_EDF },
        { kPolicyTypeLaxity                     , XSCHED_POLICY_NAME_LAX },
        // NEW_POLICY: New policy type names go here.
    };
    return kPolicyTypeNames;
}

PolicyType GetPolicyType(const std::string &name)
{
    for (auto it = PolicyTypeNames().begin(); it != PolicyTypeNames().end(); ++it) {
        if (it->second == name) return it->first;
    }
    return kPolicyTypeUnknown;
}

const std::string &GetPolicyTypeName(PolicyType type)
{
    static const std::string unk = "Unknown";
    auto it = PolicyTypeNames().find(type);
    if (it != PolicyTypeNames().end()) return it->second;
    return unk;
}

const std::map<SchedulerType, std::string> &SchedulerTypeNames() {
    static const std::map<SchedulerType, std::string> kSchedulerTypeNames {
        { kSchedulerTypeUnknown     , "Unknown"                 },
        { kSchedulerTypeLocal       , XSCHED_SCHEDULER_NAME_LCL },
        { kSchedulerTypeGlobal      , XSCHED_SCHEDULER_NAME_GLB },
        { kSchedulerTypeAppManaged  , XSCHED_SCHEDULER_NAME_AMG },
    };
    return kSchedulerTypeNames;
}

SchedulerType GetSchedulerType(const std::string &name)
{
    for (auto it = SchedulerTypeNames().begin(); it != SchedulerTypeNames().end(); ++it) {
        if (it->second == name) return it->first;
    }
    return kSchedulerTypeUnknown;
}

const std::string &GetSchedulerTypeName(SchedulerType type)
{
    static const std::string unk = "Unknown";
    auto it = SchedulerTypeNames().find(type);
    if (it != SchedulerTypeNames().end()) return it->second;
    return unk;
}

} // namespace xsched::sched
