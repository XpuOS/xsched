#include <map>
#include "xsched/protocol/def.h"
#include "xsched/sched/protocol/names.h"

namespace xsched::sched
{

const std::map<PolicyType, std::string> &PolicyTypeNames() {
    static const std::map<PolicyType, std::string> kPolicyTypeNames {
        { kPolicyTypeUnknown                    , "Unknown"              },
        { kPolicyTypeGlobal                     , XSCHED_POLICY_NAME_GBL },
        { kPolicyTypeAppManaged                 , XSCHED_POLICY_NAME_AMG },
        { kPolicyTypeHighestPriorityFirst       , XSCHED_POLICY_NAME_HPF },
        { kPolicyTypeUtilizationPartition       , XSCHED_POLICY_NAME_UP  },
        { kPolicyTypeProcessUtilizationPartition, XSCHED_POLICY_NAME_PUP },
        { kPolicyTypeEarlyDeadlineFirst         , XSCHED_POLICY_NAME_EDF },
        { kPolicyTypeLaxity                     , XSCHED_POLICY_NAME_LAX },
        { kPolicyTypeKEarlyDeadlineFirst        , XSCHED_POLICY_NAME_KEDF },
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

} // namespace xsched::sched
