#include <map>
#include "xsched/protocol/def.h"
#include "xsched/sched/protocol/names.h"

namespace xsched::sched
{

static const std::map<SchedulerType, std::string> &SchedulerNames() {
    static const std::map<SchedulerType, std::string> kSchedulerNames {
        { kSchedulerUnknown   , XSCHED_UNKNOWN_NAME       },
        { kSchedulerAppManaged, XSCHED_SCHEDULER_NAME_APP },
        { kSchedulerLocal     , XSCHED_SCHEDULER_NAME_LCL },
        { kSchedulerGlobal    , XSCHED_SCHEDULER_NAME_GLB },
    };
    return kSchedulerNames;
}

static const std::map<PolicyType, std::string> &PolicyNames() {
    static const std::map<PolicyType, std::string> kPolicyNames {
        { kPolicyUnknown                          , XSCHED_UNKNOWN_NAME     },
        { kPolicyHighestPriorityFirst             , XSCHED_POLICY_NAME_HPF  },
        { kPolicyHeterogeneousHighestPriorityFirst, XSCHED_POLICY_NAME_HHPF },
        { kPolicyUtilizationPartition             , XSCHED_POLICY_NAME_UP   },
        { kPolicyProcessUtilizationPartition      , XSCHED_POLICY_NAME_PUP  },
        { kPolicyKEarliestDeadlineFirst           , XSCHED_POLICY_NAME_KEDF },
        { kPolicyLaxity                           , XSCHED_POLICY_NAME_LAX  },
        // NEW_POLICY: New policy type names go here.
    };
    return kPolicyNames;
}

SchedulerType GetSchedulerType(const std::string &name)
{
    for (auto type : SchedulerNames()) {
        if (type.second == name) return type.first;
    }
    return kSchedulerUnknown;
}

const std::string &GetSchedulerTypeName(SchedulerType type)
{
    static const std::string unk = XSCHED_UNKNOWN_NAME;
    auto it = SchedulerNames().find(type);
    if (it != SchedulerNames().end()) return it->second;
    return unk;
}

PolicyType GetPolicyType(const std::string &name)
{
    for (auto type : PolicyNames()) {
        if (type.second == name) return type.first;
    }
    return kPolicyUnknown;
}

const std::string &GetPolicyTypeName(PolicyType type)
{
    static const std::string unk = XSCHED_UNKNOWN_NAME;
    auto it = PolicyNames().find(type);
    if (it != PolicyNames().end()) return it->second;
    return unk;
}

} // namespace xsched::sched
