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

} // namespace xsched::protocol
