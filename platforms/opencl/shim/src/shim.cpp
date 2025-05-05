#include <list>
#include <mutex>
#include <unordered_map>

#include "xsched/xqueue.h"
#include "xsched/protocol/def.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/opencl/hal/types.h"
#include "xsched/opencl/hal/driver.h"
#include "xsched/opencl/shim/shim.h"
#include "xsched/preempt/xqueue/xcommand.h"

using namespace xsched::preempt;

namespace xsched::opencl
{

static std::mutex g_args_mtx;
static std::unordered_map<cl_kernel, std::shared_ptr<std::list<KernelArgument>>> g_args;

static inline cl_int SetKernelArg(cl_kernel kernel, KernelArgumentType type, cl_uint idx, size_t size, const void *val)
{
    std::shared_ptr<std::list<KernelArgument>> args = nullptr;
    g_args_mtx.lock();
    auto it = g_args.find(kernel);
    if (it == g_args.end()) {
        args = std::make_shared<std::list<KernelArgument>>();
        g_args[kernel] = args;
    } else {
        args = it->second;
    }
    args->emplace_back(type, idx, size, val);
    g_args_mtx.unlock();
    return CL_SUCCESS;
}

cl_int XSetKernelArg(cl_kernel kernel, cl_uint idx, size_t size, const void *val)
{
    XDEBG("XSetKernelArg(kernel: %p, idx: %u, size: %zu, val: %p)", kernel, idx, size, val);
    return SetKernelArg(kernel, kArgOpenCL, idx, size, val);
}

cl_int XSetKernelArgSVMPointer(cl_kernel kernel, cl_uint idx, const void *val)
{
    XDEBG("XSetKernelArgSVMPointer(kernel: %p, idx: %u, val: %p)", kernel, idx, val);
    return SetKernelArg(kernel, kArgSVMPointer, idx, 0, val);
}

cl_int XSetKernelArgMemPointerINTEL(cl_kernel kernel, cl_uint idx, const void *val)
{
    XDEBG("XSetKernelArgMemPointerINTEL(kernel: %p, idx: %u, val: %p)", kernel, idx, val);
    return SetKernelArg(kernel, kArgMemPointerINTEL, idx, 0, val);
}

cl_int XEnqueueNDRangeKernel(cl_command_queue cmdq, cl_kernel kernel, cl_uint work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
    XDEBG("XEnqueueNDRangeKernel(cmdq: %p, kernel: %p, e_num: %u, event: %p)",
          cmdq, kernel, num_events_in_wait_list, event);

    std::shared_ptr<std::list<KernelArgument>> args = nullptr;
    g_args_mtx.lock();
    auto it = g_args.find(kernel);
    if (it != g_args.end()) {
        args = it->second;
        g_args.erase(it);
    }
    g_args_mtx.unlock();

    // OclKernelCommand::Launch() will set the kernel arguments.
    auto cmd = std::make_shared<OclKernelCommand>(
        kernel, work_dim, global_work_offset, global_work_size, local_work_size,
        num_events_in_wait_list, event_wait_list, event, args);
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(cmdq));
    if (xq == nullptr) return cmd->LaunchWrapper(cmdq);
    xq->Submit(cmd);
    return CL_SUCCESS;
}

cl_int XWaitForEvents(cl_uint num_events, const cl_event *event_list)
{
    XDEBG("XWaitForEvents(e_num: %d, e_list: %p)", num_events, event_list);
    return Driver::WaitForEvents(num_events, event_list);
}

cl_int XFinish(cl_command_queue command_queue)
{
    XDEBG("XFinish(cmdq: %p)", command_queue);
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(command_queue));
    if (xq == nullptr) return Driver::Finish(command_queue);
    xq->WaitAll();
    return CL_SUCCESS;
}

} // namespace xsched::opencl
