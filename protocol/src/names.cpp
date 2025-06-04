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

} // namespace xsched::protocol
