#pragma once

#include <string>

#include "xsched/types.h"

namespace xsched::protocol
{

XPlatform GetPlatform(const std::string &name);
const std::string &GetPlatformName(XPlatform plat);

XDeviceType GetDeviceType(const std::string &name);
const std::string &GetDeviceTypeName(XDeviceType type);

XPreemptLevel GetPreemptLevel(const std::string &name);
const std::string &GetPreemptLevelName(XPreemptLevel level);

XSchedulerType GetSchedulerType(const std::string &name);
const std::string &GetSchedulerTypeName(XSchedulerType type);

XPolicyType GetPolicyType(const std::string &name);
const std::string &GetPolicyTypeName(XPolicyType type);

} // namespace xsched::protocol
