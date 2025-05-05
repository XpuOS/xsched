#include "xsched/utils/common.h"
#include "xsched/opencl/cmdq.h"
#include "xsched/opencl/shim/shim.h"
#include "xsched/opencl/hal.h"
#include "xsched/opencl/hal/driver.h"
#include "xsched/opencl/hal/ocl_assert.h"
#include "xsched/preempt/xqueue/xqueue.h"

using namespace xsched::preempt;

namespace xsched::opencl
{

static std::mutex g_cmdq_mtx;
static int32_t g_cmdq_idx = -1;
static std::unordered_map<int32_t, cl_command_queue> g_cmdqs;

} // namespace xsched::opencl

using namespace xsched::opencl;

EXPORT_C_FUNC XResult SetOverrideCommandQueue(int32_t queue_idx)
{
    if (queue_idx < 0) return kXSchedErrorInvalidValue;
    g_cmdq_mtx.lock();
    g_cmdq_idx = queue_idx;
    g_cmdq_mtx.unlock();
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult UnsetOverrideCommandQueue()
{
    g_cmdq_mtx.lock();
    g_cmdq_idx = -1;
    g_cmdq_mtx.unlock();
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult GetOverrideCommandQueue(int32_t queue_idx, cl_command_queue *cmdq)
{
    if (cmdq == nullptr || queue_idx < 0) return kXSchedErrorInvalidValue;
    std::lock_guard<std::mutex> lock(g_cmdq_mtx);
    auto it = g_cmdqs.find(queue_idx);
    if (it == g_cmdqs.end()) return kXSchedErrorNotFound;
    *cmdq = it->second;
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult DeleteOverrideCommandQueue(int32_t queue_idx)
{
    std::lock_guard<std::mutex> lock(g_cmdq_mtx);
    auto it = g_cmdqs.find(queue_idx);
    if (it == g_cmdqs.end()) return kXSchedErrorNotFound;
    // Release the command queue to allow being deleted by the driver.
    OCL_ASSERT(Driver::ReleaseCommandQueue(it->second));
    g_cmdqs.erase(it);
    return kXSchedSuccess;
}

namespace xsched::opencl
{

cl_command_queue CreateCommandQueue(std::function<cl_command_queue()> create)
{
    std::lock_guard<std::mutex> lock(g_cmdq_mtx);
    if (g_cmdq_idx == -1) {
        auto cmdq = create();
        XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return OclQueueCreate(hwq, cmdq); });
        XDEBG("create cmdq %p", cmdq);
        return cmdq;
    }

    auto it = g_cmdqs.find(g_cmdq_idx);
    if (it != g_cmdqs.end()) {
        XDEBG("override cmdq %p", it->second);
        OCL_ASSERT(Driver::RetainCommandQueue(it->second));
        return it->second;
    }

    auto cmdq = create();
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return OclQueueCreate(hwq, cmdq); });
    XDEBG("create & override cmdq %p", cmdq);
    if (cmdq == nullptr) return nullptr;
    g_cmdqs[g_cmdq_idx] = cmdq;
    // One more retain the cmdq to avoid being deleted by the driver.
    OCL_ASSERT(Driver::RetainCommandQueue(cmdq));
    return cmdq;
}

cl_command_queue XCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int *errcode_ret)
{
    XDEBG("XCreateCommandQueue(ctx: %p, dev: %p, prop: 0x%lx)", context, device, properties);
    return CreateCommandQueue([&]() {
        return Driver::CreateCommandQueue(context, device, properties, errcode_ret);
    });
}

cl_command_queue XCreateCommandQueueWithProperties(cl_context context, cl_device_id device, const cl_queue_properties *properties, cl_int *errcode_ret)
{
    XDEBG("XCreateCommandQueueWithProperties(ctx: %p, dev: %p)", context, device);
    return CreateCommandQueue([&]() {
        return Driver::CreateCommandQueueWithProperties(context, device, properties, errcode_ret);
    });
}

} // namespace xsched::opencl
