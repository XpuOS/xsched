#pragma once

#include <string>

#include "xsched/types.h"

namespace xsched::protocol
{

XDeviceId GetDeviceId(XDevice dev);
XDeviceType GetDeviceType(XDevice dev);
XDevice MakeDevice(XDeviceType type, XDeviceId id);

} // namespace xsched::protocol
