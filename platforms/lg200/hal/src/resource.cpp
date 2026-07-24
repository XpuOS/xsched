#include "xsched/lg200/hal/resource.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "xsched/utils/common.h"
#include "xsched/utils/log.h"
#include "xsched/utils/pci.h"

namespace fs = std::filesystem;

namespace xsched::lg200
{

#if defined(__linux__)

static fs::path DrmRoot()
{
    const char *override_path = std::getenv("XSCHED_LG200_DRM_ROOT");
    return override_path == nullptr || override_path[0] == '\0'
        ? fs::path("/sys/class/drm")
        : fs::path(override_path);
}

static std::string ReadTextFile(const fs::path &path)
{
    std::ifstream in(path);
    if (!in) return "";

    std::ostringstream ss;
    ss << in.rdbuf();
    auto text = ss.str();
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.pop_back();
    }
    return text;
}

static std::string Lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

static void CopyField(char *dst, size_t dst_size, const std::string &src)
{
    if (dst_size == 0) return;
    std::strncpy(dst, src.c_str(), dst_size - 1);
    dst[dst_size - 1] = '\0';
}

static uint32_t ParseCardIndex(const std::string &name)
{
    if (name.rfind("card", 0) != 0) return UINT32_MAX;
    const std::string number = name.substr(4);
    if (number.empty()) return UINT32_MAX;

    uint32_t value = 0;
    for (unsigned char character : number) {
        if (character < static_cast<unsigned char>('0') || character > static_cast<unsigned char>('9')) {
            return UINT32_MAX;
        }
        const uint32_t digit = character - static_cast<unsigned char>('0');
        if (value > (UINT32_MAX - digit) / 10U) return UINT32_MAX;
        value = value * 10U + digit;
    }
    return value;
}

static bool ParsePciId(const std::string &pci_name, XDeviceId *device_id)
{
    unsigned int domain = 0;
    unsigned int bus = 0;
    unsigned int device = 0;
    unsigned int function = 0;

    if (std::sscanf(pci_name.c_str(), "%x:%x:%x.%x",
            &domain, &bus, &device, &function)
        != 4) {
        return false;
    }

    *device_id = MakePciId(domain, bus, device, function);
    return true;
}

static std::string ReadUeventValue(const fs::path &device_path, const std::string &key)
{
    const std::string content = ReadTextFile(device_path / "uevent");
    std::istringstream lines(content);
    std::string line;
    const std::string prefix = key + "=";

    while (std::getline(lines, line)) {
        if (line.rfind(prefix, 0) == 0) return line.substr(prefix.size());
    }

    return "";
}

static uint64_t ReadMemoryBytes(const fs::path &device_path)
{
    const std::string value = ReadTextFile(device_path / "mem_info_vram_total");
    if (value.empty()) return 0;
    try {
        return std::stoull(value);
    } catch (...) {
        return 0;
    }
}

static uint64_t ReadUint64File(const fs::path &path)
{
    const std::string value = ReadTextFile(path);
    if (value.empty()) return 0;
    try {
        return std::stoull(value);
    } catch (...) {
        return 0;
    }
}

static double ReadDoubleFile(const fs::path &path)
{
    const std::string value = ReadTextFile(path);
    if (value.empty()) return -1.0;
    try {
        return std::stod(value);
    } catch (...) {
        return -1.0;
    }
}

static bool IsLg200Candidate(const std::string &vendor,
    const std::string &driver,
    const std::string &name)
{
    const std::string text = Lower(vendor + " " + driver + " " + name);
    const std::string normalized_driver = Lower(driver);
    // The Loongson PCI vendor id alone is not a GPU identity: other display
    // or bridge devices can use the same vendor. Require either the LG200
    // product name or the vendor kernel driver's explicit loonggpu identity.
    return text.find("lg200") != std::string::npos || normalized_driver.rfind("loonggpu", 0) == 0;
}

#endif

std::vector<Lg200DeviceInfo> DiscoverDevices()
{
    std::vector<Lg200DeviceInfo> devices;

#if defined(__linux__)
    const fs::path drm_root = DrmRoot();
    if (!fs::exists(drm_root)) return devices;

    for (const auto &entry : fs::directory_iterator(drm_root)) {
        const std::string card_name = entry.path().filename().string();
        const uint32_t card_index = ParseCardIndex(card_name);
        if (card_index == UINT32_MAX) continue;

        const fs::path device_path = entry.path() / "device";
        if (!fs::exists(device_path)) continue;

        const std::string vendor = ReadTextFile(device_path / "vendor");
        const std::string driver = ReadUeventValue(device_path, "DRIVER");
        const std::string pci_id = ReadUeventValue(device_path, "PCI_ID");
        const std::string model = pci_id.empty() ? card_name : pci_id;
        if (!IsLg200Candidate(vendor, driver, model)) continue;

        Lg200DeviceInfo info { };
        info.drm_card = card_index;
        info.local_memory_bytes = ReadMemoryBytes(device_path);
        CopyField(info.name, sizeof(info.name), model);
        CopyField(info.vendor, sizeof(info.vendor), vendor);
        CopyField(info.driver, sizeof(info.driver), driver);

        std::error_code ec;
        const fs::path real_device = fs::canonical(device_path, ec);
        if (!ec) ParsePciId(real_device.filename().string(), &info.device_id);

        devices.push_back(info);
    }
    std::sort(devices.begin(), devices.end(), [](const auto &left, const auto &right) {
        return left.drm_card < right.drm_card;
    });
#endif

    return devices;
}

} // namespace xsched::lg200

EXPORT_C_FUNC XResult Lg200DeviceEnumerate(Lg200DeviceInfo *devices, uint32_t *count)
{
    if (count == nullptr) return kXSchedErrorInvalidValue;

    std::vector<Lg200DeviceInfo> discovered;
    try {
        discovered = xsched::lg200::DiscoverDevices();
    } catch (const std::exception &error) {
        XWARN("LG200 device discovery failed: %s", error.what());
        return kXSchedErrorUnknown;
    }

    const uint32_t required = static_cast<uint32_t>(discovered.size());
    if (devices == nullptr) {
        *count = required;
        return kXSchedSuccess;
    }

    const uint32_t capacity = *count;
    const uint32_t copied = std::min(capacity, required);
    for (uint32_t i = 0; i < copied; ++i) {
        devices[i] = discovered[i];
    }

    *count = required;
    return capacity < required ? kXSchedErrorInvalidValue : kXSchedSuccess;
}

EXPORT_C_FUNC XResult Lg200DeviceReadStats(uint32_t drm_card, Lg200DeviceStats *stats)
{
    if (stats == nullptr) return kXSchedErrorInvalidValue;

    *stats = { };
    stats->drm_card = drm_card;
    stats->gpu_utilization = -1.0;

#if defined(__linux__)
    try {
        const auto devices = xsched::lg200::DiscoverDevices();
        const bool discovered = std::any_of(
            devices.begin(), devices.end(), [drm_card](const Lg200DeviceInfo &device) {
                return device.drm_card == drm_card;
            });
        if (!discovered) return kXSchedErrorNotFound;
    } catch (const std::exception &error) {
        XWARN("LG200 device validation failed: %s", error.what());
        return kXSchedErrorUnknown;
    }

    const fs::path device_path = xsched::lg200::DrmRoot() / ("card" + std::to_string(drm_card)) / "device";
    std::error_code ec;
    if (!fs::exists(device_path, ec) || ec) return kXSchedErrorNotFound;

    stats->gpu_utilization = xsched::lg200::ReadDoubleFile(device_path / "gpu_busy_percent");
    stats->local_memory_used_bytes = xsched::lg200::ReadUint64File(device_path / "mem_info_vram_used");
    stats->local_memory_total_bytes = xsched::lg200::ReadUint64File(device_path / "mem_info_vram_total");
    return kXSchedSuccess;
#else
    return kXSchedErrorNotSupported;
#endif
}
