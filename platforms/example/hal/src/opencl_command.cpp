#include "xsched/opencl/hal/opencl_command.h"

using namespace xsched::opencl;

OpenclKernelCommand::OpenclKernelCommand(cl_kernel kernel, cl_uint work_dim,
                                         const size_t *global_work_offset,
                                         const size_t *global_work_size,
                                         const size_t *local_work_size,
                                         cl_uint num_events_in_wait_list,
                                         const cl_event *event_wait_list,
                                         cl_event *event,
                                         size_t num_args, KernelArgument *args)
    : kernel_(kernel), work_dim_(work_dim),
      num_events_in_wait_list_(num_events_in_wait_list), event_wait_list_(event_wait_list),
      event_(event), num_args_(num_args)
{
    if (work_dim_ <= 0) return;
    
    const size_t buf_size = work_dim_ * sizeof(size_t);
    global_work_size_ = (size_t *)malloc(buf_size);
    memcpy(global_work_size_, global_work_size, buf_size);

    // local_work_size is optional and can be nullptr
    if (local_work_size != nullptr) {
        local_work_size_ = (size_t *)malloc(buf_size);
        memcpy(local_work_size_, local_work_size, buf_size);
    }

    // global_work_offset is optional and can be nullptr
    if (global_work_offset != nullptr) {
        global_work_offset_ = (size_t *)malloc(buf_size);
        memcpy(global_work_offset_, global_work_offset, buf_size);
    }

    if (num_args_ <= 0) return;
    args_ = (KernelArgument *)malloc(num_args_ * sizeof(KernelArgument));
    for (size_t i = 0; i < num_args_; i++) {
        args_[i].index = args[i].index;
        args_[i].size = args[i].size;
        args_[i].value = nullptr;
        if (args[i].size == 0 || args[i].value == nullptr) continue;
        args_[i].value = malloc(args[i].size);
        memcpy(args_[i].value, args[i].value, args[i].size);
    }
}

OpenclKernelCommand::~OpenclKernelCommand()
{
    if (global_work_offset_ != nullptr) free(global_work_offset_);
    if (global_work_size_ != nullptr) free(global_work_size_);
    if (local_work_size_ != nullptr) free(local_work_size_);
    if (num_args_ > 0 && args_ != nullptr) {
        for (size_t i = 0; i < num_args_; i++) {
            if (args_[i].value != nullptr) free(args_[i].value);
        }
        free(args_);
    }
}

cl_int OpenclKernelCommand::Launch(cl_command_queue cmdq)
{
    for (size_t i = 0; i < num_args_; i++) {
        auto ret = Driver::SetKernelArg(kernel_, args_[i].index, args_[i].size, args_[i].value);
        if (ret != CL_SUCCESS) return ret;
    }
    return Driver::EnqueueNDRangeKernel(cmdq, kernel_, work_dim_,
                                        global_work_offset_,
                                        global_work_size_,
                                        local_work_size_,
                                        num_events_in_wait_list_, event_wait_list_, event_);
}

OpenclReadCommand::OpenclReadCommand(cl_mem buffer, cl_bool blocking_read,
                                     size_t offset, size_t size, void *ptr,
                                     cl_uint num_events_in_wait_list,
                                     const cl_event *event_wait_list, cl_event *event)
    : buffer_(buffer), blocking_read_(blocking_read), offset_(offset), size_(size), ptr_(ptr)
    , num_events_in_wait_list_(num_events_in_wait_list)
    , event_wait_list_(event_wait_list), event_(event) {}

cl_int OpenclReadCommand::Launch(cl_command_queue cmdq)
{
    return Driver::EnqueueReadBuffer(cmdq, buffer_, blocking_read_, offset_, size_, ptr_,
                                     num_events_in_wait_list_, event_wait_list_, event_);
}

OpenclWriteCommand::OpenclWriteCommand(cl_mem buffer, cl_bool blocking_write,
                                       size_t offset, size_t size, const void *ptr,
                                       cl_uint num_events_in_wait_list,
                                       const cl_event *event_wait_list, cl_event *event)
    : buffer_(buffer), blocking_write_(blocking_write), offset_(offset), size_(size), ptr_(ptr)
    , num_events_in_wait_list_(num_events_in_wait_list)
    , event_wait_list_(event_wait_list), event_(event) {}

cl_int OpenclWriteCommand::Launch(cl_command_queue cmdq)
{
    return Driver::EnqueueWriteBuffer(cmdq, buffer_, blocking_write_, offset_, size_, ptr_,
                                      num_events_in_wait_list_, event_wait_list_, event_);
}

XResult OpenclKernelCommandCreate(HwCommandHandle *hw_cmd,
                                  cl_kernel kernel, cl_uint work_dim,
                                  const size_t *global_work_offset,
                                  const size_t *global_work_size,
                                  const size_t *local_work_size,
                                  cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                                  cl_event *event,
                                  size_t num_args, KernelArgument *args)
{
    auto cmd = std::make_shared<OpenclKernelCommand>(kernel, work_dim, global_work_offset,
                                                     global_work_size, local_work_size,
                                                     num_events_in_wait_list, event_wait_list,
                                                     event, num_args, args);
    *hw_cmd = xsched::preempt::HwCommandManager::Add(cmd);
    return kXSchedSuccess;
}

XResult OpenclReadCommandCreate(HwCommandHandle *hw_cmd,
                                cl_mem buffer, cl_bool blocking_read,
                                size_t offset, size_t size, void *ptr,
                                cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                                cl_event *event)
{
    auto cmd = std::make_shared<OpenclReadCommand>(buffer, blocking_read, offset, size, ptr,
                                                   num_events_in_wait_list, event_wait_list,
                                                   event);
    *hw_cmd = xsched::preempt::HwCommandManager::Add(cmd);
    return kXSchedSuccess;
}

XResult OpenclWriteCommandCreate(HwCommandHandle *hw_cmd,
                                 cl_mem buffer, cl_bool blocking_write,
                                 size_t offset, size_t size, const void *ptr,
                                 cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                                 cl_event *event)
{
    auto cmd = std::make_shared<OpenclWriteCommand>(buffer, blocking_write, offset, size, ptr,
                                                    num_events_in_wait_list, event_wait_list,
                                                    event);
    *hw_cmd = xsched::preempt::HwCommandManager::Add(cmd);
    return kXSchedSuccess;
}
