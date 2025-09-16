#include <iostream>
#include <level_zero/ze_api.h>
#include "ze_driver_info.h"

void printDriverInfo(const ze_driver_properties_t &props)
{
    std::cout << "Driver Version: "
              << ZE_MAJOR_VERSION(props.driverVersion) << "."
              << ZE_MINOR_VERSION(props.driverVersion) << std::endl;
    std::cout << "Driver UUID: ";
    for (int i = 0; i < ZE_MAX_DRIVER_UUID_SIZE; ++i)
    {
        printf("%02x", props.uuid.id[i]);
    }
    std::cout << std::endl;
}

void printDeviceInfo(const ze_device_properties_t &props)
{
    std::cout << "Device Name: " << props.name << std::endl;
    std::cout << "Device Type: ";
    switch (props.type)
    {
    case ZE_DEVICE_TYPE_GPU:
        std::cout << "GPU";
        break;
    case ZE_DEVICE_TYPE_CPU:
        std::cout << "CPU";
        break;
    case ZE_DEVICE_TYPE_FPGA:
        std::cout << "FPGA";
        break;
    case ZE_DEVICE_TYPE_MCA:
        std::cout << "MCA";
        break;
    case ZE_DEVICE_TYPE_VPU:
        std::cout << "VPU";
        break;
    default:
        std::cout << "Unknown";
    }
    std::cout << std::endl;
    std::cout << "Device Vendor ID: " << props.vendorId << std::endl;
    std::cout << "Device Core Clock Rate: " << props.coreClockRate << " MHz" << std::endl;
    std::cout << "Max Memory Allocation Size: " << props.maxMemAllocSize / (1024 * 1024) << " MB" << std::endl;
}

void PrintZEDriverInfo()
{
    ze_result_t result;

    ZE_ASSERT(zeInit(ZE_INIT_FLAG_FORCE_UINT32));

    uint32_t driverCount = 0;
    ZE_ASSERT(zeDriverGet(&driverCount, nullptr));
    std::cout << "Found " << driverCount << " Level Zero driver(s)" << std::endl;

    ze_driver_handle_t *drivers = new ze_driver_handle_t[driverCount];
    ZE_ASSERT(zeDriverGet(&driverCount, drivers));

    for (uint32_t i = 0; i < driverCount; ++i)
    {
        std::cout << "\nDriver #" << i << ":" << std::endl;

        ze_driver_properties_t driverProps = {};
        driverProps.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
        ZE_ASSERT(zeDriverGetProperties(drivers[i], &driverProps));
        printDriverInfo(driverProps);

        ze_api_version_t apiVersion;
        ZE_ASSERT(zeDriverGetApiVersion(drivers[i], &apiVersion));

        std::cout << "Level Zero API Version: "
                  << ZE_MAJOR_VERSION(apiVersion) << "."
                  << ZE_MINOR_VERSION(apiVersion) << std::endl;

        uint32_t deviceCount = 0;
        ZE_ASSERT(zeDeviceGet(drivers[i], &deviceCount, nullptr));
        if (deviceCount == 0)
        {
            printf("Number of device is 0\n");
            continue;
        }

        ze_device_handle_t *devices = new ze_device_handle_t[deviceCount];
        ZE_ASSERT(zeDeviceGet(drivers[i], &deviceCount, devices));

        for (uint32_t j = 0; j < deviceCount; ++j)
        {
            std::cout << "\nDevice #" << j << ":" << std::endl;

            ze_device_properties_t deviceProps = {};
            deviceProps.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ZE_ASSERT(zeDeviceGetProperties(devices[j], &deviceProps));
            printDeviceInfo(deviceProps);
        }

        delete[] devices;
    }

    delete[] drivers;
    return;
}