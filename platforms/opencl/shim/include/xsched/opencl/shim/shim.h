#pragma once

#include <memory>

#include "xsched/utils/function.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/opencl/hal/types.h"
#include "xsched/opencl/hal/driver.h"
#include "xsched/opencl/hal/ocl_command.h"

namespace xsched::opencl
{

#define OPENCL_SHIM_FUNC(name, cmd, ...) \
inline cl_int X##name(cl_command_queue cmdq __VA_OPT__(,) FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__)) \
{ \
    auto xq = xsched::preempt::HwQueueManager::GetXQueue(GetHwQueueHandle(cmdq)); \
    if (xq == nullptr) return Driver::name(cmdq __VA_OPT__(,) FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__)); \
    auto hw_cmd = std::make_shared<cmd>(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__)); \
    xq->Submit(hw_cmd); \
    return CL_SUCCESS; \
}

////////////////////////////// kernel related //////////////////////////////
cl_int XSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void * arg_value);
cl_int XSetKernelArgSVMPointer(cl_kernel kernel, cl_uint arg_index, const void *arg_value);
cl_int XSetKernelArgMemPointerINTEL(cl_kernel kernel, cl_uint arg_index, const void *arg_value);
cl_int XEnqueueNDRangeKernel(cl_command_queue cmdq, cl_kernel kernel, cl_uint work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);

////////////////////////////// memory related //////////////////////////////
OPENCL_SHIM_FUNC(EnqueueReadBuffer, OclReadBufferCommand, cl_mem, buffer, cl_bool, blocking_read, size_t, offset, size_t, size, void *, ptr, cl_uint, num_events_in_wait_list, const cl_event *, event_wait_list, cl_event *, event);
OPENCL_SHIM_FUNC(EnqueueWriteBuffer, OclWriteBufferCommand, cl_mem, buffer, cl_bool, blocking_write, size_t, offset, size_t, size, const void *, ptr, cl_uint, num_events_in_wait_list, const cl_event *, event_wait_list, cl_event *, event);
OPENCL_SHIM_FUNC(EnqueueFillBuffer, OclFillBufferCommand, cl_mem, buffer, const void *, pattern, size_t, pattern_size, size_t, offset, size_t, size, cl_uint, num_events_in_wait_list, const cl_event *, event_wait_list, cl_event *, event);
OPENCL_SHIM_FUNC(EnqueueCopyBuffer, OclCopyBufferCommand, cl_mem, src_buffer, cl_mem, dst_buffer, size_t, src_offset, size_t, dst_offset, size_t, size, cl_uint, num_events_in_wait_list, const cl_event *, event_wait_list, cl_event *, event);
OPENCL_SHIM_FUNC(EnqueueMemFillINTEL, OclMemFillINTELCommand, void *, dst_ptr, const void *, pattern, size_t, pattern_size, size_t, size, cl_uint, num_events_in_wait_list, const cl_event *, event_wait_list, cl_event *, event);
OPENCL_SHIM_FUNC(EnqueueMemcpyINTEL, OclMemcpyINTELCommand, cl_bool, blocking, void *, dst_ptr, const void *, src_ptr, size_t, size, cl_uint, num_events_in_wait_list, const cl_event *, event_wait_list, cl_event *, event);
OPENCL_SHIM_FUNC(EnqueueMemsetINTEL, OclMemsetINTELCommand, void *, dst_ptr, cl_int, value, size_t, size, cl_uint, num_events_in_wait_list, const cl_event *, event_wait_list, cl_event *, event);

////////////////////////////// event related //////////////////////////////
cl_int XWaitForEvents(cl_uint num_events, const cl_event *event_list);
OPENCL_SHIM_FUNC(EnqueueBarrierWithWaitList, OclBarrierCommand, cl_uint, num_events_in_wait_list, const cl_event *, event_wait_list, cl_event *, event);

////////////////////////////// command queue related //////////////////////////////
OPENCL_SHIM_FUNC(Flush, OclFlushCommand);
cl_int XFinish(cl_command_queue command_queue);

cl_command_queue XCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int *errcode_ret);
cl_command_queue XCreateCommandQueueWithProperties(cl_context context, cl_device_id device, const cl_queue_properties *properties, cl_int *errcode_ret);

} // namespace xsched::opencl
