
#pragma once

#include "xsched/opencl/hal.h"
#include "xsched/opencl/hal/driver.h"
#include "xsched/preempt/hal/hw_command.h"

namespace xsched::opencl
{

class OpenclCommand : public preempt::HwCommand
{
public:
    OpenclCommand() = default;
    virtual ~OpenclCommand() = default;

    virtual void Synchronize() override {}
    virtual bool Synchronizable() override { return false; }
    virtual bool EnableSynchronization() override { return false; }
    virtual cl_int Launch(cl_command_queue cmdq) = 0;
};

class OpenclKernelCommand : public OpenclCommand
{
public:
    OpenclKernelCommand(cl_kernel kernel, cl_uint work_dim,
                        const size_t *global_work_offset,
                        const size_t *global_work_size,
                        const size_t *local_work_size,
                        cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                        cl_event *event,
                        size_t num_args, KernelArgument *args);
    virtual ~OpenclKernelCommand();
    virtual cl_int Launch(cl_command_queue cmdq) override;

private:
    cl_kernel kernel_;
    cl_uint work_dim_;
    size_t *global_work_offset_ = nullptr;
    size_t *global_work_size_ = nullptr;
    size_t *local_work_size_ = nullptr;
    cl_uint num_events_in_wait_list_;
    const cl_event *event_wait_list_ = nullptr;
    cl_event *event_ = nullptr;
    size_t num_args_;
    KernelArgument *args_ = nullptr;
};

class OpenclReadCommand : public OpenclCommand
{
public:
    OpenclReadCommand(cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void *ptr,
                      cl_uint num_events_in_wait_list,
                      const cl_event *event_wait_list, cl_event *event);
    virtual ~OpenclReadCommand() = default;
    virtual cl_int Launch(cl_command_queue cmdq) override;

private:
    cl_mem buffer_;
    cl_bool blocking_read_;
    size_t offset_;
    size_t size_;
    void *ptr_;
    cl_uint num_events_in_wait_list_;
    const cl_event *event_wait_list_ = nullptr;
    cl_event *event_ = nullptr;
};

class OpenclWriteCommand : public OpenclCommand
{
public:
    OpenclWriteCommand(cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void *ptr,
                       cl_uint num_events_in_wait_list,
                       const cl_event *event_wait_list, cl_event *event);
    virtual ~OpenclWriteCommand() = default;
    virtual cl_int Launch(cl_command_queue cmdq) override;

private:
    cl_mem buffer_;
    cl_bool blocking_write_;
    size_t offset_;
    size_t size_;
    const void *ptr_;
    cl_uint num_events_in_wait_list_;
    const cl_event *event_wait_list_ = nullptr;
    cl_event *event_ = nullptr;
};

}  // namespace xsched::opencl
