#pragma once

#include <queue>
#include <mutex>

#include "xsched/preempt/hal/hw_command.h"
#include "xsched/hip/hal/driver.h"
#include "xsched/hip/hal/hip_assert.h"

namespace xsched::hip
{

#define HIP_COMMAND(name, base, func, ...) \
    class name : public base \
    { \
    public: \
        name(FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__)) \
            : base(), FOR_EACH_PAIR_COMMA(DECLARE_COPY_PRIVATE_ARG, __VA_ARGS__) {} \
        virtual ~name() = default; \
    private: \
        FOR_EACH_PAIR_SEMICOLON(DECLARE_PRIVATE_PARAM, __VA_ARGS__) \
        virtual hipError_t Launch(hipStream_t stream) override \
        { return func(FOR_EACH_PAIR_COMMA(DECLARE_PRIVATE_ARG, __VA_ARGS__), stream); } \
    };

class HipCommand : public preempt::HwCommand
{
public:
    HipCommand(): HwCommand(preempt::kCommandPropertyNone) {}
    virtual ~HipCommand();

    virtual void Synchronize() override;
    virtual bool Synchronizable() override;
    virtual bool EnableSynchronization() override;
    hipError_t LaunchWrapper(hipStream_t stream);

private:
    virtual hipError_t Launch(hipStream_t stream) = 0;
    hipEvent_t following_event_ = nullptr;
};

// HIP has many kernel launch commands, including 
// 1. hipLaunchKernel
// 2. hipLaunchKernelEx
// 3. hipModuleLaunchKernel
// 4. hipModuleLaunchCooperativeKernel
// 
// Moreover, these commands accept different kinds of "functions".
// For hipLaunchKernel and hipLaunchKernelEx, the function is a const void * (static function)
// For hipModuleLaunchKernel and others, the function is a hipFunction_t (dynamic function)
// 
// These two types of functions requires different parameter copying methods.

class HipKernelCommand : public HipCommand
{
public:
    HipKernelCommand(void** kernel_params, void **extra_params, bool copy_param)
        : param_copied_(copy_param)
        , original_kernel_params_(kernel_params), original_extra_params_(extra_params) {}
    virtual ~HipKernelCommand();

protected:
    size_t param_cnt_ = 0;
    size_t param_buffer_size_ = 0;
    char *param_data_ = nullptr;
    void **kernel_params_ = nullptr; 
    bool param_copied_ = false; // synchronous calls don't need copying
    void **original_kernel_params_ = nullptr;
    void **original_extra_params_ = nullptr;
};

// Kernels from static code object
class HipStaticKernelLaunchCommand : public HipKernelCommand
{
public:
    HipStaticKernelLaunchCommand(const void *host_func, void **params, void **extra, bool copy);
    virtual ~HipStaticKernelLaunchCommand() = default;
protected:
    const void * const host_func_;
};

class HipDynamicKernelLaunchCommand : public HipKernelCommand
{
public:
    HipDynamicKernelLaunchCommand(hipFunction_t function, void **params, void **extra, bool copy);
    virtual ~HipDynamicKernelLaunchCommand() = default;
protected:
    const hipFunction_t function_;
};

class HipKernelLaunchCommand : public HipStaticKernelLaunchCommand
{
public:
    HipKernelLaunchCommand(const void *host_func, dim3 num_blocks, dim3 block_dim,
                           void **params, unsigned int shm, bool copy_param)
        : HipStaticKernelLaunchCommand(host_func, params, nullptr, copy_param)
        , num_blocks_(num_blocks), block_dim_(block_dim), shared_mem_bytes_(shm) {}
    virtual ~HipKernelLaunchCommand() = default;

private:
    virtual hipError_t Launch(hipStream_t stream) override;
    const dim3 num_blocks_;
    const dim3 block_dim_;
    const unsigned int shared_mem_bytes_;
};

class HipModuleKernelLaunchCommand : public HipDynamicKernelLaunchCommand
{
public:
    HipModuleKernelLaunchCommand(hipFunction_t function,
                                 unsigned int gdx, unsigned int gdy, unsigned int gdz,
                                 unsigned int bdx, unsigned int bdy, unsigned int bdz,
                                 unsigned int shared_mem_bytes,
                                 void **params, void **extra, bool copy)
        : HipDynamicKernelLaunchCommand(function, params, extra, copy)
        , gdx_(gdx), gdy_(gdy), gdz_(gdz)
        , bdx_(bdx), bdy_(bdy), bdz_(bdz)
        , shared_mem_bytes_(shared_mem_bytes) {}
    virtual ~HipModuleKernelLaunchCommand() = default;

private:
    virtual hipError_t Launch(hipStream_t stream) override;
    const unsigned int gdx_, gdy_, gdz_;
    const unsigned int bdx_, bdy_, bdz_;
    const unsigned int shared_mem_bytes_;
};


class HipExtModuleKernelLaunchCommand : public HipDynamicKernelLaunchCommand
{
public:
    HipExtModuleKernelLaunchCommand(hipFunction_t function, 
                                    uint32_t gwx, uint32_t gwy, uint32_t gwz,
                                    uint32_t lwx, uint32_t lwy, uint32_t lwz,
                                    size_t shm, void **params, void **extra,
                                    hipEvent_t start_event, hipEvent_t stop_event, 
                                    uint32_t flags, bool copy_param)
        : HipDynamicKernelLaunchCommand(function, params, extra, copy_param)
        , gwx_(gwx), gwy_(gwy), gwz_(gwz), lwx_(lwx), lwy_(lwy), lwz_(lwz)
        , shm_(shm), start_event_(start_event), stop_event_(stop_event), flags_(flags) {}
    virtual ~HipExtModuleKernelLaunchCommand() = default;

private:
    virtual hipError_t Launch(hipStream_t stream) override;
    const uint32_t gwx_, gwy_, gwz_;
    const uint32_t lwx_, lwy_, lwz_;
    const size_t shm_;
    const hipEvent_t start_event_;
    const hipEvent_t stop_event_;
    const uint32_t flags_;
};

class HipHostFuncCommand : public HipCommand
{
public:
    HipHostFuncCommand(hipHostFn_t fn, void *user_data): fn_(fn), user_data_(user_data) {}
    virtual ~HipHostFuncCommand() = default;

private:
    const hipHostFn_t fn_;
    void * const user_data_;
    virtual hipError_t Launch(hipStream_t stream) override
    { return Driver::LaunchHostFunc(stream, fn_, user_data_); }
};

class HipMemoryCommand : public HipCommand
{
public:
    HipMemoryCommand() {}
    virtual ~HipMemoryCommand() = default;
};

class HipMemcpyWithStreamCommand : public HipMemoryCommand
{
public:
    HipMemcpyWithStreamCommand(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind)
        : dst_(dst), src_(src), sizeBytes_(sizeBytes), kind_(kind) {}
    virtual ~HipMemcpyWithStreamCommand() = default;

private:
    virtual hipError_t Launch(hipStream_t stream) override;
    void *dst_;
    const void *src_;
    size_t sizeBytes_;
    hipMemcpyKind kind_;
};

HIP_COMMAND(HipMemcpyHtoDCommand, HipMemoryCommand, Driver::MemcpyHtoDAsync, hipDeviceptr_t , dst_device, void *, src_host, size_t, byte_count);
HIP_COMMAND(HipMemcpyDtoHCommand, HipMemoryCommand, Driver::MemcpyDtoHAsync, void *, dst_host, hipDeviceptr_t, src_device, size_t, byte_count);
HIP_COMMAND(HipMemcpyDtoDCommand, HipMemoryCommand, Driver::MemcpyDtoDAsync, hipDeviceptr_t, dst_device, hipDeviceptr_t, src_device, size_t, byte_count);
HIP_COMMAND(HipMemcpyAsyncCommand, HipMemoryCommand, Driver::MemcpyAsync, void *, dst, const void *, src, size_t, sizeBytes, hipMemcpyKind, kind);
HIP_COMMAND(HipMemsetAsyncCommand, HipMemoryCommand, Driver::MemsetAsync, void *, dst, int, value, size_t, sizeBytes);
HIP_COMMAND(HipMemsetD8Command, HipMemoryCommand, Driver::MemsetD8Async, hipDeviceptr_t, dst_device, unsigned char, value, size_t, n);
HIP_COMMAND(HipMemsetD16Command, HipMemoryCommand, Driver::MemsetD16Async, hipDeviceptr_t, dst_device, unsigned short, value, size_t, n);
HIP_COMMAND(HipMemsetD32Command, HipMemoryCommand, Driver::MemsetD32Async, hipDeviceptr_t, dst_device, unsigned int, value, size_t, n);

class HipEventRecordCommand : public HipCommand
{
public:
    HipEventRecordCommand(hipEvent_t event);
    virtual ~HipEventRecordCommand();

    virtual void Synchronize() override { HIP_ASSERT(Driver::EventSynchronize(event_)); }
    virtual bool Synchronizable() override { return true; }
    virtual bool EnableSynchronization() override { return true; }

    // Mark the event_ as destroyed, so that the event_ will be destroyed in the destructor.
    void DestroyEvent() { destroy_event_ = true; }

protected:
    hipEvent_t event_;

private:
    virtual hipError_t Launch(hipStream_t stream) override
    {
        XDEBG("HipEventRecordCommand(%p): stream = %p, event = %p", this, stream, event_);
        return Driver::EventRecord(event_, stream);
    }
    bool destroy_event_ = false;
};

class HipEventRecordWithFlagsCommand : public HipEventRecordCommand
{
public:
    HipEventRecordWithFlagsCommand(hipEvent_t event, unsigned int flags)
        : HipEventRecordCommand(event), flags_(flags) {}
    virtual ~HipEventRecordWithFlagsCommand() = default;

private:
    virtual hipError_t Launch(hipStream_t stream) override
    {
        XDEBG("HipEventRecordWithFlagsCommand(%p): stream = %p, event = %p, flags = %d", this, stream, event_, flags_);
        return Driver::EventRecord(event_, stream);
    }
    const unsigned int flags_;
};

class HipEventWaitCommand : public HipCommand
{
public:
    HipEventWaitCommand(hipEvent_t event, unsigned int flags)
        : event_(event), event_record_command_(nullptr), flags_(flags) {}
    HipEventWaitCommand(std::shared_ptr<HipEventRecordCommand> event_cmd, unsigned int flags)
        : event_(nullptr), event_record_command_(event_cmd), flags_(flags) {}
    virtual ~HipEventWaitCommand() = default;

    virtual void BeforeLaunch() override;

private:
    virtual hipError_t Launch(hipStream_t stream) override;
    const hipEvent_t event_;
    const std::shared_ptr<HipEventRecordCommand> event_record_command_;
    const unsigned int flags_;
};

} // namespace xsched::hal::hip
