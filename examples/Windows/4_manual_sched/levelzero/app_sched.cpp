#include <level_zero/ze_api.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <thread>
#include <chrono>
#include <random>
#include <cstdio>
#include <cstdlib>

#include "xsched/xsched.h"
#include "xsched/levelzero/hal.h"

// 1 is for low priority, 2 is for high priority
ze_command_list_handle_t cmdl_1, cmdl_2;
ze_command_queue_handle_t cmdq_1, cmdq_2;
HwQueueHandle hwq_1, hwq_2;
XQueueHandle xq_1, xq_2;

// Global variable
ze_context_handle_t context;
ze_device_handle_t device;
ze_module_handle_t module;
const size_t N = 1 << 24;                                       // Vector size, 16 MB
const int M = 1000;                                             // Number of tasks loop
const int K = 10;                                               // Number of vector additions per task
const uint32_t groupSizeX = 64;                                 // Number of threads per workgroup
const uint32_t groupCountX = (N + groupSizeX - 1) / groupSizeX; // Number of groups activated

#define ZE_ASSERT(cmd)                                                            \
    do                                                                            \
    {                                                                             \
        ze_result_t res = cmd;                                                    \
        if (res != ZE_RESULT_SUCCESS)                                             \
        {                                                                         \
            printf("Level Zero error: 0x%x at %s:%d\n", res, __FILE__, __LINE__); \
            exit(1);                                                              \
        }                                                                         \
    } while (0);

std::vector<uint8_t> loadSPIRV(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open SPIR-V file: " + filename);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (!file.read((char *)buffer.data(), size))
    {
        throw std::runtime_error("Failed to read SPIR-V file");
    }
    printf("Finish loading SPIRV file\n");
    return buffer;
}

void run(bool is_high_priority)
{
    // Allocate memory: Use shared memory (visible to host device)
    float *src_a = nullptr;
    float *src_b = nullptr;
    float *src_c = nullptr;

    ze_device_mem_alloc_desc_t deviceDesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    ze_host_mem_alloc_desc_t hostDesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};

    ZE_ASSERT(zeMemAllocShared(context, &deviceDesc, &hostDesc, N * sizeof(float), sizeof(float), device, (void **)&src_a));
    ZE_ASSERT(zeMemAllocShared(context, &deviceDesc, &hostDesc, N * sizeof(float), sizeof(float), device, (void **)&src_b));
    ZE_ASSERT(zeMemAllocShared(context, &deviceDesc, &hostDesc, N * sizeof(float), sizeof(float), device, (void **)&src_c));

    // Initialized Data
    for (size_t i = 0; i < N; ++i)
    {
        src_a[i] = static_cast<float>(rand()) / RAND_MAX;
        src_b[i] = static_cast<float>(rand()) / RAND_MAX;
        src_c[i] = 0;
    }

    // Create kernel
    ze_kernel_handle_t kernel;
    ze_kernel_desc_t kernelDesc = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        nullptr,
        0,           // flags
        "vector_add" // name of kernel
    };
    ZE_ASSERT(zeKernelCreate(module, &kernelDesc, &kernel));
    ZE_ASSERT(zeKernelSetGroupSize(kernel, groupSizeX, 1, 1));

    // Set kernel parameters
    ZE_ASSERT(zeKernelSetArgumentValue(kernel, 0, sizeof(src_a), &src_a));
    ZE_ASSERT(zeKernelSetArgumentValue(kernel, 1, sizeof(src_b), &src_b));
    ZE_ASSERT(zeKernelSetArgumentValue(kernel, 2, sizeof(src_c), &src_c));
    ZE_ASSERT(zeKernelSetArgumentValue(kernel, 3, sizeof(N), &N));

    ze_group_count_t dispatchTraits = {groupCountX, 1, 1};
    if (is_high_priority)
    {
        ZE_ASSERT(zeCommandListAppendLaunchKernel(cmdl_2, kernel, &dispatchTraits, nullptr, 0, nullptr));
        ZE_ASSERT(zeCommandListClose(cmdl_2));
    }
    else
    {
        ZE_ASSERT(zeCommandListAppendLaunchKernel(cmdl_1, kernel, &dispatchTraits, nullptr, 0, nullptr));
        ZE_ASSERT(zeCommandListClose(cmdl_1));
    }

    // Execute
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(30, 50);
    for (int i = 0; i < M; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < K; ++i)
        {
            if (is_high_priority)
            {
                // If this is the high-priority task,
                // suspend the low-priority task when the high-priority task starts.
                XQueueSuspend(xq_1, 0);
                ZE_ASSERT(zeCommandQueueExecuteCommandLists(cmdq_2, 1, &cmdl_2, nullptr));
                ZE_ASSERT(zeCommandQueueSynchronize(cmdq_2, UINT32_MAX));
                XQueueResume(xq_1, 0);
            }
            else
            {
                ZE_ASSERT(zeCommandQueueExecuteCommandLists(cmdq_1, 1, &cmdl_1, nullptr));
                ZE_ASSERT(zeCommandQueueSynchronize(cmdq_1, UINT32_MAX));
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("%s prio Task %d completed in %lld ms\n", is_high_priority ? "high" : "low ", i, duration.count());

        // Verify results
        // bool passed = true;
        // for (size_t i = 0; i < N; ++i)
        // {
        //     float expected = src_a[i] + src_b[i];
        //     if (std::abs(src_c[i] - expected) > 1e-5f)
        //     {
        //         std::cout << "Mismatch at " << i << ": got " << src_c[i] << ", expected " << expected << std::endl;
        //         passed = false;
        //         break;
        //     }
        // }

        // Sleep for random interval between tasks
        // std::this_thread::sleep_for(std::chrono::milliseconds(dis(gen)));
    }

    // Clean
    ZE_ASSERT(zeMemFree(context, src_a));
    ZE_ASSERT(zeMemFree(context, src_b));
    ZE_ASSERT(zeMemFree(context, src_c));
    ZE_ASSERT(zeKernelDestroy(kernel));
    ZE_ASSERT(zeModuleDestroy(module));
    ZE_ASSERT(zeContextDestroy(context));

    if (is_high_priority)
    {
        ZE_ASSERT(zeCommandListDestroy(cmdl_2));
        ZE_ASSERT(zeCommandQueueDestroy(cmdq_2));
    }
    else
    {
        ZE_ASSERT(zeCommandListDestroy(cmdl_1));
        ZE_ASSERT(zeCommandQueueDestroy(cmdq_1));
    }
}

int main()
{
    // Init
    ZE_ASSERT(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    // Get driver
    uint32_t driverCount = 0;
    ZE_ASSERT(zeDriverGet(&driverCount, nullptr));
    std::vector<ze_driver_handle_t> drivers(driverCount);
    ZE_ASSERT(zeDriverGet(&driverCount, drivers.data()));

    // Create context
    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    ZE_ASSERT(zeContextCreate(drivers[0], &contextDesc, &context));

    // Get device (GPU)
    uint32_t deviceCount = 0;
    ZE_ASSERT(zeDeviceGet(drivers[0], &deviceCount, nullptr));
    std::vector<ze_device_handle_t> devices(deviceCount);
    ZE_ASSERT(zeDeviceGet(drivers[0], &deviceCount, devices.data()));
    device = devices[0];

    // Create command queue and command list
    ze_command_queue_desc_t queueDesc_1 = {
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        nullptr,
        0,
        0,
        0,
        ZE_COMMAND_QUEUE_MODE_DEFAULT,
        ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_queue_desc_t queueDesc_2 = {
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        nullptr,
        1,
        0,
        0,
        ZE_COMMAND_QUEUE_MODE_DEFAULT,
        ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ZE_ASSERT(zeCommandQueueCreate(context, device, &queueDesc_1, &cmdq_1));
    ZE_ASSERT(zeCommandQueueCreate(context, device, &queueDesc_2, &cmdq_2));

    ZeQueueCreate(&hwq_1, device, cmdq_1);
    ZeQueueCreate(&hwq_2, device, cmdq_2);
    XQueueCreate(&xq_1, hwq_1, kPreemptLevelBlock, kQueueCreateFlagNone);
    XQueueCreate(&xq_2, hwq_2, kPreemptLevelBlock, kQueueCreateFlagNone);

    ze_command_list_desc_t cmdListDesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
    ZE_ASSERT(zeCommandListCreate(context, device, &cmdListDesc, &cmdl_1));
    ZE_ASSERT(zeCommandListCreate(context, device, &cmdListDesc, &cmdl_2));
    printf("Finish initialization\n");

    // Load SPIR-V
    std::vector<uint8_t> spirv = loadSPIRV("vector_add.spv");

    // Create module
    ze_module_desc_t moduleDesc = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        nullptr,
        ZE_MODULE_FORMAT_IL_SPIRV,
        spirv.size(),
        spirv.data(),
        "", // compiler flags
        nullptr};
    ZE_ASSERT(zeModuleCreate(context, device, &moduleDesc, &module, nullptr));

    // run two tasks within one process
    std::thread thread_lp(run, false);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::thread thread_hp(run, true);

    thread_lp.join();
    thread_hp.join();

    return 0;
}