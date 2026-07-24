#pragma once

#include <vector>

#include "xsched/lg200/hal.h"

namespace xsched::lg200
{

std::vector<Lg200DeviceInfo> DiscoverDevices();

} // namespace xsched::lg200
