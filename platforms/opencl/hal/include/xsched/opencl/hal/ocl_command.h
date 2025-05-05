#pragma once

#include <list>
#include <memory>

#include "xsched/utils/function.h"
#include "xsched/opencl/hal/driver.h"
#include "xsched/preempt/hal/hw_command.h"

namespace xsched::opencl
{

#define OCL_COMMAND(name, base, func, blocking, ...) \
    class name : public base \
    { \
    public: \
        name(FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__) __VA_OPT__(,) \
             cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) \
            : base(num_events_in_wait_list, event_wait_list, event) __VA_OPT__(,) \
              FOR_EACH_PAIR_COMMA(DECLARE_COPY_PRIVATE_ARG, __VA_ARGS__) \
        { if (blocking == CL_TRUE) this->SetProps(preempt::kCommandPropertyBlockingSubmit); } \
        virtual ~name() = default; \
    private: \
        FOR_EACH_PAIR_SEMICOLON(DECLARE_PRIVATE_PARAM, __VA_ARGS__) \
        virtual cl_int Launch(cl_command_queue cmdq, cl_uint num_wait, const cl_event *wait_list, \
                              cl_event *event) override \
        { \
            return func(cmdq, FOR_EACH_PAIR_COMMA(DECLARE_PRIVATE_ARG, __VA_ARGS__) __VA_OPT__(,) \
                        num_wait, wait_list, event); \
        } \
    };

class OclCommand : public preempt::HwCommand
{
public:
    OclCommand(cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
    virtual ~OclCommand();

    cl_int LaunchWrapper(cl_command_queue cmdq);
    virtual void Synchronize() override;
    virtual bool Synchronizable() override;
    virtual bool EnableSynchronization() override;

private:
    virtual cl_int Launch(cl_command_queue cmdq, cl_uint num_wait, const cl_event *wait_list,
                          cl_event *event) = 0;

    bool sync_enabled_ = false;
    cl_event following_event_ = nullptr;

    const cl_uint num_events_in_wait_list_;
    cl_event *event_wait_list_ = nullptr;
    cl_event *user_event_ptr_ = nullptr;
};

enum KernelArgumentType
{
    kArgOpenCL          = 0,
    kArgSVMPointer      = 1,
    kArgMemPointerINTEL = 2,
};

class KernelArgument
{
public:
    NO_COPY_CLASS(KernelArgument);
    NO_MOVE_CLASS(KernelArgument);

    KernelArgument(KernelArgumentType type, cl_uint index, size_t size, const void *value);
    ~KernelArgument();
    void Set(cl_kernel kernel) const;

private:
    const KernelArgumentType type_;
    const cl_uint index_;
    const size_t size_;
    const void *value_ = nullptr;
};

class OclKernelCommand : public OclCommand
{
public:
    OclKernelCommand(cl_kernel kernel, cl_uint work_dim,
                     const size_t *global_work_offset,
                     const size_t *global_work_size,
                     const size_t *local_work_size,
                     cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                     cl_event *event,
                     std::shared_ptr<std::list<KernelArgument>> args);
    virtual ~OclKernelCommand();

private:
    virtual cl_int Launch(cl_command_queue cmdq, cl_uint num_wait, const cl_event *wait_list,
                          cl_event *event) override;

    const cl_kernel kernel_;
    const cl_uint work_dim_;
    size_t *global_work_offset_ = nullptr;
    size_t *global_work_size_ = nullptr;
    size_t *local_work_size_ = nullptr;
    std::shared_ptr<std::list<KernelArgument>> args_ = nullptr;
};

class OclFlushCommand : public OclCommand
{
public:
    OclFlushCommand(): OclCommand(0, nullptr, nullptr)
    { this->SetProps(preempt::kCommandPropertyBlockingSubmit); }
    virtual ~OclFlushCommand() = default;
    virtual bool Synchronizable() override { return false; }
    virtual bool EnableSynchronization() override { return false; }

private:
    virtual cl_int Launch(cl_command_queue cmdq, cl_uint, const cl_event *, cl_event *) override
    { return Driver::Flush(cmdq); }
};

OCL_COMMAND(OclBarrierCommand, OclCommand, Driver::EnqueueBarrierWithWaitList, CL_FALSE);
OCL_COMMAND(OclReadBufferCommand, OclCommand, Driver::EnqueueReadBuffer, blocking_read, cl_mem, buffer, cl_bool, blocking_read, size_t, offset, size_t, size, void *, ptr);
OCL_COMMAND(OclWriteBufferCommand, OclCommand, Driver::EnqueueWriteBuffer, blocking_write, cl_mem, buffer, cl_bool, blocking_write, size_t, offset, size_t, size, const void *, ptr);
OCL_COMMAND(OclFillBufferCommand, OclCommand, Driver::EnqueueFillBuffer, CL_FALSE, cl_mem, buffer, const void *, pattern, size_t, pattern_size, size_t, offset, size_t, size);
OCL_COMMAND(OclCopyBufferCommand, OclCommand, Driver::EnqueueCopyBuffer, CL_FALSE, cl_mem, src_buffer, cl_mem, dst_buffer, size_t, src_offset, size_t, dst_offset, size_t, size);
OCL_COMMAND(OclMemFillINTELCommand, OclCommand, Driver::EnqueueMemFillINTEL, CL_FALSE, void *, dst_ptr, const void *, pattern, size_t, pattern_size, size_t, size);
OCL_COMMAND(OclMemcpyINTELCommand, OclCommand, Driver::EnqueueMemcpyINTEL, blocking, cl_bool, blocking, void *, dst_ptr, const void *, src_ptr, size_t, size);
OCL_COMMAND(OclMemsetINTELCommand, OclCommand, Driver::EnqueueMemsetINTEL, CL_FALSE, void *, dst_ptr, cl_int, value, size_t, size);

} // namespace xsched::opencl
