#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "xsched/lg200/hal.h"

static const char *OrUnknown(const char *value)
{
    return value[0] == '\0' ? "unknown" : value;
}

static std::string FormatBytes(uint64_t bytes)
{
    if (bytes == 0) return "unknown";

    const char *units[] = { "B", "KiB", "MiB", "GiB" };
    double value = static_cast<double>(bytes);
    size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < std::size(units)) {
        value /= 1024.0;
        ++unit;
    }

    std::ostringstream out;
    out << std::fixed << std::setprecision(unit == 0 ? 0 : 2) << value << ' ' << units[unit];
    return out.str();
}

static std::string FormatUtilization(double value)
{
    if (value < 0.0) return "unknown";

    std::ostringstream out;
    out << std::fixed << std::setprecision(1) << value << '%';
    return out.str();
}

int main(int argc, char **argv)
{
    bool require_device = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--require-device") {
            require_device = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: lg200_info [--require-device]\n";
            return 0;
        } else {
            std::cerr << "unknown option: " << arg << '\n';
            return 2;
        }
    }

    uint32_t count = 0;
    XResult res = Lg200DeviceEnumerate(nullptr, &count);
    if (res != kXSchedSuccess) {
        std::cerr << "failed to enumerate LG200 devices: " << res << '\n';
        return 1;
    }

    if (count == 0) {
        std::cout << "No LG200 DRM devices found.\n";
        return require_device ? 3 : 0;
    }

    std::vector<Lg200DeviceInfo> devices(count);
    res = Lg200DeviceEnumerate(devices.data(), &count);
    if (res != kXSchedSuccess) {
        std::cerr << "failed to read LG200 device info: " << res << '\n';
        return 1;
    }

    std::cout << "LG200 devices: " << count << '\n';
    for (uint32_t i = 0; i < count; ++i) {
        const auto &dev = devices[i];
        Lg200DeviceStats stats { };
        const bool has_stats = Lg200DeviceReadStats(dev.drm_card, &stats) == kXSchedSuccess;
        std::cout << "  [" << i << "] drm=card" << dev.drm_card
                  << " device_id=0x" << std::hex << dev.device_id << std::dec
                  << " name=" << OrUnknown(dev.name)
                  << " vendor=" << OrUnknown(dev.vendor)
                  << " driver=" << OrUnknown(dev.driver)
                  << " local_memory=" << FormatBytes(dev.local_memory_bytes)
                  << " gpu_utilization="
                  << (has_stats ? FormatUtilization(stats.gpu_utilization) : "unknown")
                  << " memory_used="
                  << (has_stats ? FormatBytes(stats.local_memory_used_bytes) : "unknown")
                  << '\n';
    }

    return 0;
}
