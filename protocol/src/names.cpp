#include <map>
#include "xsched/protocol/def.h"
#include "xsched/protocol/names.h"

namespace xsched::protocol
{

static const std::map<XPlatform, const std::string> &PlatformNames() {
    static const std::map<XPlatform, const std::string> kPlatformNames {
        { kPlatformUnknown  , XSCHED_UNKNOWN_NAME },
        { kPlatformVPI      , "VPI"       },
        { kPlatformCUDA     , "CUDA"      },
        { kPlatformCUDLA    , "cuDLA"     },
        { kPlatformHIP      , "HIP"       },
        { kPlatformAscend   , "Ascend"    },
        { kPlatformOpenCL   , "OpenCL"    },
        { kPlatformLevelZero, "LevelZero" },
        // NEW_PLATFORM: New platform names go here.
    };
    return kPlatformNames;
}

static const std::map<XDeviceType, std::string> &DeviceTypeNames() {
    static const std::map<XDeviceType, std::string> kDeviceTypeNames {
        { kDeviceTypeUnknown, XSCHED_UNKNOWN_NAME },
        { kDeviceTypeCPU    , "CPU"  },
        { kDeviceTypeGPU    , "GPU"  },
        { kDeviceTypeNPU    , "NPU"  },
        { kDeviceTypeFPGA   , "FPGA" },
        { kDeviceTypeASIC   , "ASIC" },
    };
    return kDeviceTypeNames;
}

static const std::map<XPreemptLevel, std::string> &PreemptLevelNames() {
    static const std::map<XPreemptLevel, std::string> kPreemptLevelNames {
        { kPreemptLevelUnknown   , XSCHED_UNKNOWN_NAME },
        { kPreemptLevelBlock     , "Block"      },
        { kPreemptLevelDeactivate, "Deactivate" },
        { kPreemptLevelInterrupt , "Interrupt"  },
    };
    return kPreemptLevelNames;
}

static const std::map<XSchedulerType, std::string> &SchedulerNames() {
    static const std::map<XSchedulerType, std::string> kSchedulerNames {
        { kSchedulerUnknown   , XSCHED_UNKNOWN_NAME       },
        { kSchedulerAppManaged, XSCHED_SCHEDULER_NAME_APP },
        { kSchedulerLocal     , XSCHED_SCHEDULER_NAME_LCL },
        { kSchedulerGlobal    , XSCHED_SCHEDULER_NAME_GLB },
    };
    return kSchedulerNames;
}

static const std::map<XPolicyType, std::string> &PolicyNames() {
    static const std::map<XPolicyType, std::string> kPolicyNames {
        { kPolicyUnknown                          , XSCHED_UNKNOWN_NAME     },
        { kPolicyHighestPriorityFirst             , XSCHED_POLICY_NAME_HPF  },
        { kPolicyHeterogeneousHighestPriorityFirst, XSCHED_POLICY_NAME_HHPF },
        { kPolicyUtilizationPartition             , XSCHED_POLICY_NAME_UP   },
        { kPolicyProcessUtilizationPartition      , XSCHED_POLICY_NAME_PUP  },
        { kPolicyKEarliestDeadlineFirst           , XSCHED_POLICY_NAME_KEDF },
        { kPolicyLaxity                           , XSCHED_POLICY_NAME_LAX  },
        { kPolicyActiveWindowFirst                , XSCHED_POLICY_NAME_AWF  },
        // NEW_POLICY: New policy type names go here.
    };
    return kPolicyNames;
}

#define SEARCH_NAME(map, key) \
    static const std::string unk = XSCHED_UNKNOWN_NAME; \
    auto it = map.find(key);                  \
    if (it != map.end()) return it->second;   \
    return unk;

#define SEARCH_KEY(map, val, unk) \
    for (auto it = map.begin(); it != map.end(); ++it) { \
        if (it->second == val) return it->first;         \
    }                                                    \
    return unk;

XPlatform GetPlatform(const std::string &name)
{
    SEARCH_KEY(PlatformNames(), name, kPlatformUnknown);
}

const std::string &GetPlatformName(XPlatform plat)
{
    SEARCH_NAME(PlatformNames(), plat);
}

XDeviceType GetDeviceType(const std::string &name)
{
    SEARCH_KEY(DeviceTypeNames(), name, kDeviceTypeUnknown);
}

const std::string &GetDeviceTypeName(XDeviceType type)
{
    SEARCH_NAME(DeviceTypeNames(), type);
}

XPreemptLevel GetPreemptLevel(const std::string &name)
{
    SEARCH_KEY(PreemptLevelNames(), name, kPreemptLevelUnknown);
}

const std::string &GetPreemptLevelName(XPreemptLevel level)
{
    SEARCH_NAME(PreemptLevelNames(), level);
}

XSchedulerType GetSchedulerType(const std::string &name)
{
    SEARCH_KEY(SchedulerNames(), name, kSchedulerUnknown);
}

const std::string &GetSchedulerTypeName(XSchedulerType type)
{
    SEARCH_NAME(SchedulerNames(), type);
}

XPolicyType GetPolicyType(const std::string &name)
{
    SEARCH_KEY(PolicyNames(), name, kPolicyUnknown);
}

const std::string &GetPolicyTypeName(XPolicyType type)
{
    SEARCH_NAME(PolicyNames(), type);
}

} // namespace xsched::protocol
