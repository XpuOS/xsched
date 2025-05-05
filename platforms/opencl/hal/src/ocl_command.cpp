#include "xsched/opencl/hal/ocl_assert.h"
#include "xsched/opencl/hal/ocl_command.h"

using namespace xsched::opencl;
using namespace xsched::preempt;

OclCommand::OclCommand(cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                       cl_event *event)
    : preempt::HwCommand(kCommandPropertyNone)
    , num_events_in_wait_list_(num_events_in_wait_list), user_event_ptr_(event)
{
    if (num_events_in_wait_list_ > 0) {
        const size_t buf_size = num_events_in_wait_list_ * sizeof(cl_event);
        event_wait_list_ = (cl_event *)malloc(buf_size);
        XASSERT(event_wait_list_ != nullptr, "failed to malloc event_wait_list_");
        memcpy(event_wait_list_, event_wait_list, buf_size);
    }

    if (user_event_ptr_ != nullptr) {
        sync_enabled_ = true;
        // we need to return the event to the application, 
        // so the submission must wait until the command is actually launched
        this->SetProps(kCommandPropertyBlockingSubmit);
    }
}

OclCommand::~OclCommand()
{
    if (event_wait_list_ != nullptr) free(event_wait_list_);
    if (following_event_ != nullptr) OCL_ASSERT(Driver::ReleaseEvent(following_event_));
}

cl_int OclCommand::LaunchWrapper(cl_command_queue cmdq)
{
    // TODO: wait until all events in event_wait_list_ are launched
    cl_int ret = Launch(cmdq, num_events_in_wait_list_, event_wait_list_,
                        sync_enabled_ ? &following_event_ : nullptr);
    if (sync_enabled_) OCL_ASSERT(Driver::Flush(cmdq));
    if (user_event_ptr_ != nullptr) {
        OCL_ASSERT(Driver::RetainEvent(following_event_));
        *user_event_ptr_ = following_event_;
    }
    return ret;
}

void OclCommand::Synchronize()
{
    XASSERT(sync_enabled_, "synchronization is not enabled");
    XASSERT(following_event_ != nullptr, "following event is not set");
    OCL_ASSERT(Driver::WaitForEvents(1, &following_event_));
}

bool OclCommand::Synchronizable()
{
    return sync_enabled_;
}

bool OclCommand::EnableSynchronization()
{
    sync_enabled_ = true;
    return true;
}

KernelArgument::KernelArgument(KernelArgumentType t, cl_uint idx, size_t sz, const void *val)
    : type_(t), index_(idx), size_(sz), value_(val)
{
    if (size_ == 0) return;
    void *ptr = malloc(size_);
    XASSERT(ptr != nullptr, "failed to allocate memory for argument");
    memcpy(ptr, val, size_);
    value_ = ptr;
}

KernelArgument::~KernelArgument()
{
    if (size_ == 0 || value_ == nullptr) return;
    free((void *)value_);
}

void KernelArgument::Set(cl_kernel kernel) const
{
    switch (type_) {
    case kArgOpenCL:
        OCL_ASSERT(Driver::SetKernelArg(kernel, index_, size_, value_));
        break;
    case kArgSVMPointer:
        OCL_ASSERT(Driver::SetKernelArgSVMPointer(kernel, index_, value_));
        break;
    case kArgMemPointerINTEL:
        OCL_ASSERT(Driver::SetKernelArgMemPointerINTEL(kernel, index_, value_));
        break;
    default:
        XASSERT(false, "invalid argument type: %d", type_);
    }
}

OclKernelCommand::OclKernelCommand(cl_kernel kernel, cl_uint work_dim,
                                   const size_t *global_work_offset,
                                   const size_t *global_work_size,
                                   const size_t *local_work_size,
                                   cl_uint num_events_in_wait_list,
                                   const cl_event *event_wait_list,
                                   cl_event *event,
                                   std::shared_ptr<std::list<KernelArgument>> args)
    : OclCommand(num_events_in_wait_list, event_wait_list, event)
    , kernel_(kernel) , work_dim_(work_dim), args_(args)
{
    if (work_dim_ <= 0) return;
    
    const size_t buf_size = work_dim_ * sizeof(size_t);
    global_work_size_ = (size_t *)malloc(buf_size);
    XASSERT(global_work_size_ != nullptr, "failed to malloc global_work_size_");
    XASSERT(global_work_size != nullptr, "global_work_size is nullptr");
    memcpy(global_work_size_, global_work_size, buf_size);

    // local_work_size is optional and can be nullptr
    if (local_work_size != nullptr) {
        local_work_size_ = (size_t *)malloc(buf_size);
        XASSERT(local_work_size_ != nullptr, "failed to malloc local_work_size_");
        memcpy(local_work_size_, local_work_size, buf_size);
    }

    // global_work_offset is optional and can be nullptr
    if (global_work_offset != nullptr) {
        global_work_offset_ = (size_t *)malloc(buf_size);
        XASSERT(global_work_offset_ != nullptr, "failed to malloc global_work_offset_");
        memcpy(global_work_offset_, global_work_offset, buf_size);
    }
}

OclKernelCommand::~OclKernelCommand()
{
    if (global_work_offset_ != nullptr) free(global_work_offset_);
    if (global_work_size_ != nullptr) free(global_work_size_);
    if (local_work_size_ != nullptr) free(local_work_size_);
}

cl_int OclKernelCommand::Launch(cl_command_queue cmdq, cl_uint num_wait, const cl_event *wait_list,
                                cl_event *event)
{
    // TODO: acquire the lock for to set args, because the kernel may be used by multiple threads
    if (args_ != nullptr) { for (const auto &arg : *args_) arg.Set(kernel_); }
    return Driver::EnqueueNDRangeKernel(cmdq, kernel_, work_dim_,
                                        global_work_offset_,
                                        global_work_size_,
                                        local_work_size_,
                                        num_wait, wait_list, event);
}
