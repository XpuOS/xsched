#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

#include "xsched/lg200/hal.h"

namespace fs = std::filesystem;

namespace
{

void WriteFile(const fs::path &path, const std::string &contents)
{
    std::ofstream output(path);
    output << contents;
    if (!output) throw std::runtime_error("failed to write " + path.string());
}

bool Check(bool condition, const char *message)
{
    if (condition) return true;
    std::cerr << "FAIL: " << message << '\n';
    return false;
}

} // namespace

int main()
{
    const fs::path root = fs::temp_directory_path() / ("xsched-lg200-resource-" + std::to_string(getpid()));
    std::error_code ec;
    fs::remove_all(root, ec);

    try {
        const fs::path device = root / "devices" / "0000:01:02.0";
        fs::create_directories(device);
        fs::create_directories(root / "card3");
        fs::create_directory_symlink(device, root / "card3" / "device");

        const fs::path non_gpu = root / "devices" / "0000:02:00.0";
        fs::create_directories(non_gpu);
        fs::create_directories(root / "card1");
        fs::create_directory_symlink(non_gpu, root / "card1" / "device");

        WriteFile(device / "vendor", "0x0014\n");
        WriteFile(device / "uevent", "DRIVER=loonggpu\nPCI_ID=0014:7a25\n");
        WriteFile(device / "gpu_busy_percent", "73.5\n");
        WriteFile(device / "mem_info_vram_used", "1048576\n");
        WriteFile(device / "mem_info_vram_total", "8388608\n");
        WriteFile(non_gpu / "vendor", "0x0014\n");
        WriteFile(non_gpu / "uevent", "DRIVER=othergpu\nPCI_ID=0014:1234\n");

        if (setenv("XSCHED_LG200_DRM_ROOT", root.string().c_str(), 1) != 0) {
            throw std::runtime_error("setenv failed");
        }

        bool ok = true;
        std::uint32_t count = 0;
        ok &= Check(Lg200DeviceEnumerate(nullptr, &count) == kXSchedSuccess,
            "count query failed");
        ok &= Check(count == 1, "vendor-only non-LG200 device must be rejected");

        std::vector<Lg200DeviceInfo> devices(count);
        std::uint32_t capacity = count;
        ok &= Check(Lg200DeviceEnumerate(devices.data(), &capacity) == kXSchedSuccess,
            "device enumeration failed");
        ok &= Check(capacity == 1, "enumeration returned wrong count");
        ok &= Check(devices[0].drm_card == 3, "wrong DRM card index");
        ok &= Check(std::string(devices[0].driver) == "loonggpu", "wrong driver name");
        ok &= Check(std::string(devices[0].vendor) == "0x0014", "wrong vendor id");
        ok &= Check(devices[0].local_memory_bytes == 8388608, "wrong total memory");

        Lg200DeviceInfo dummy { };
        capacity = 0;
        ok &= Check(Lg200DeviceEnumerate(&dummy, &capacity) == kXSchedErrorInvalidValue,
            "short buffer must fail");
        ok &= Check(capacity == 1, "short buffer must report required capacity");
        ok &= Check(Lg200DeviceEnumerate(nullptr, nullptr) == kXSchedErrorInvalidValue,
            "null count must fail");

        Lg200DeviceStats stats { };
        ok &= Check(Lg200DeviceReadStats(3, &stats) == kXSchedSuccess,
            "stat query failed");
        ok &= Check(stats.gpu_utilization == 73.5, "wrong GPU utilization");
        ok &= Check(stats.local_memory_used_bytes == 1048576, "wrong used memory");
        ok &= Check(stats.local_memory_total_bytes == 8388608, "wrong stat total memory");
        ok &= Check(Lg200DeviceReadStats(1, &stats) == kXSchedErrorNotFound,
            "non-LG200 card stats must return not found");
        ok &= Check(Lg200DeviceReadStats(99, &stats) == kXSchedErrorNotFound,
            "missing card must return not found");

        unsetenv("XSCHED_LG200_DRM_ROOT");
        fs::remove_all(root, ec);
        if (!ok) return 1;
        std::cout << "LG200 resource test passed.\n";
        return 0;
    } catch (const std::exception &error) {
        unsetenv("XSCHED_LG200_DRM_ROOT");
        fs::remove_all(root, ec);
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
