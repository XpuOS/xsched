#pragma once

#include "xsched/preempt/hal/hw_command.h"
#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

namespace xsched::cuda
{

#define CUDA_COMMAND(name, base, func, ...) \
    class name : public base \
    { \
    public: \
        name(FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__)) \
            : base(), FOR_EACH_PAIR_COMMA(DECLARE_COPY_PRIVATE_ARG, __VA_ARGS__) {} \
        virtual ~name() = default; \
    private: \
        FOR_EACH_PAIR_SEMICOLON(DECLARE_PRIVATE_PARAM, __VA_ARGS__) \
        virtual CUresult Launch(CUstream stream) override \
        { return func(FOR_EACH_PAIR_COMMA(DECLARE_PRIVATE_ARG, __VA_ARGS__), stream); } \
    };

class CudaCommand : public preempt::HwCommand
{
public:
    CudaCommand(preempt::XCommandProperties props);
    virtual ~CudaCommand();
    virtual void Synchronize() override;
    virtual bool Synchronizable() override;
    virtual bool EnableSynchronization() override;
    CUresult LaunchWrapper(CUstream stream);

private:
    CUcontext ctx_ = nullptr;
    CUevent following_event_ = nullptr;
    virtual CUresult Launch(CUstream stream) = 0;
};

// cuda kernels
class CudaKernelCommand : public CudaCommand
{
public:
    CudaKernelCommand(CUfunction func, void **params, void **extra, bool deep_copy);
    virtual ~CudaKernelCommand();
    virtual unsigned int BlockCnt() const = 0;

    const CUfunction kFunc;
    bool killable = false;
    CUdeviceptr preempt_buffer = 0;
    CUdeviceptr entry_point_original = 0;
    CUdeviceptr entry_point_instrumented = 0;

protected:
    bool deep_copy_ = false;
    void **params_ = nullptr;
    void ** const extra_;

private:
    size_t param_cnt_ = 0;
    char *param_data_ = nullptr;
};

class CudaKernelLaunchCommand : public CudaKernelCommand
{
public:
    CudaKernelLaunchCommand(CUfunction func,
                            unsigned int gdx, unsigned int gdy, unsigned int gdz,
                            unsigned int bdx, unsigned int bdy, unsigned int bdz,
                            unsigned int shm, void **params, void **extra, bool deep_copy)
        : CudaKernelCommand(func, params, extra, deep_copy)
        , gdx_(gdx), gdy_(gdy), gdz_(gdz), bdx_(bdx), bdy_(bdy), bdz_(bdz), shm_(shm) {}
    virtual ~CudaKernelLaunchCommand() = default;
    virtual unsigned int BlockCnt() const override { return gdx_ * gdy_ * gdz_; }

private:
    const unsigned int gdx_, gdy_, gdz_; // grid dimension
    const unsigned int bdx_, bdy_, bdz_; // block dimension
    const unsigned int shm_; // shared memory byte size
    virtual CUresult Launch(CUstream stream) override
    {
        return Driver::LaunchKernel(kFunc, gdx_, gdy_, gdz_, bdx_, bdy_, bdz_,
                                    shm_, stream, params_, extra_);
    }
};

class CudaKernelLaunchExCommand : public CudaKernelCommand
{
public:
    CudaKernelLaunchExCommand(const CUlaunchConfig *cfg, CUfunction func,
                              void **params, void **extra, bool deep_copy);
    virtual ~CudaKernelLaunchExCommand();
    virtual unsigned int BlockCnt() const override
    { return cfg_.gridDimX * cfg_.gridDimY * cfg_.gridDimZ; }

private:
    CUlaunchConfig cfg_;
    virtual CUresult Launch(CUstream stream) override;
};

// host function
class CudaHostFuncCommand : public CudaCommand
{
public:
    CudaHostFuncCommand(CUhostFn fn, void *data)
        : CudaCommand(preempt::kCommandPropertyNone), fn_(fn), data_(data) {}
    virtual ~CudaHostFuncCommand() = default;

private:
    const CUhostFn fn_;
    void * const data_;
    virtual CUresult Launch(CUstream stream) override
    { return Driver::LaunchHostFunc(stream, fn_, data_); }
};

// memory commands
class CudaMemoryCommand : public CudaCommand
{
public:
    CudaMemoryCommand(): CudaCommand(preempt::kCommandPropertyNone) {}
    virtual ~CudaMemoryCommand() = default;
};

CUDA_COMMAND(CudaMemcpyHtoDV2Command, CudaMemoryCommand, Driver::MemcpyHtoDAsync_v2, CUdeviceptr, dstDevice, const void *, srcHost, size_t, ByteCount);
CUDA_COMMAND(CudaMemcpyDtoHV2Command, CudaMemoryCommand, Driver::MemcpyDtoHAsync_v2, void *, dstHost, CUdeviceptr, srcDevice, size_t, ByteCount);
CUDA_COMMAND(CudaMemcpyDtoDV2Command, CudaMemoryCommand, Driver::MemcpyDtoDAsync_v2, CUdeviceptr, dstDevice, CUdeviceptr, srcDevice, size_t, ByteCount);
CUDA_COMMAND(CudaMemsetD8Command, CudaMemoryCommand, Driver::MemsetD8Async, CUdeviceptr, dstDevice, unsigned char, uc, size_t, N);
CUDA_COMMAND(CudaMemsetD16Command, CudaMemoryCommand, Driver::MemsetD16Async, CUdeviceptr, dstDevice, unsigned short, us, size_t, N);
CUDA_COMMAND(CudaMemsetD32Command, CudaMemoryCommand, Driver::MemsetD32Async, CUdeviceptr, dstDevice, unsigned int, ui, size_t, N);
CUDA_COMMAND(CudaMemsetD2D8Command, CudaMemoryCommand, Driver::MemsetD2D8Async, CUdeviceptr, dstDevice, size_t, dstPitch, unsigned char, uc, size_t, Width, size_t, Height);
CUDA_COMMAND(CudaMemsetD2D16Command, CudaMemoryCommand, Driver::MemsetD2D16Async, CUdeviceptr, dstDevice, size_t, dstPitch, unsigned short, us, size_t, Width, size_t, Height);
CUDA_COMMAND(CudaMemsetD2D32Command, CudaMemoryCommand, Driver::MemsetD2D32Async, CUdeviceptr, dstDevice, size_t, dstPitch, unsigned int, ui, size_t, Width, size_t, Height);
CUDA_COMMAND(CudaMemoryAllocCommand, CudaMemoryCommand, Driver::MemAllocAsync, CUdeviceptr *, dptr, size_t, bytesize);
CUDA_COMMAND(CudaMemoryFreeCommand, CudaMemoryCommand, Driver::MemFreeAsync, CUdeviceptr, dptr);

class CudaMemcpy2DV2Command : public CudaMemoryCommand
{
public:
    CudaMemcpy2DV2Command(const CUDA_MEMCPY2D *p_copy): CudaMemoryCommand(), copy_(*p_copy) {}
    virtual ~CudaMemcpy2DV2Command() = default;
private:
    const CUDA_MEMCPY2D copy_;
    virtual CUresult Launch(CUstream stream) override
    { return Driver::Memcpy2DAsync_v2(&copy_, stream); }
};

class CudaMemcpy3DV2Command : public CudaMemoryCommand
{
public:
    CudaMemcpy3DV2Command(const CUDA_MEMCPY3D *p_copy): CudaMemoryCommand(), copy_(*p_copy) {}
    virtual ~CudaMemcpy3DV2Command() = default;
private:
    const CUDA_MEMCPY3D copy_;
    virtual CUresult Launch(CUstream stream) override
    { return Driver::Memcpy3DAsync_v2(&copy_, stream); }
};

// cuda events
class CudaEventRecordCommand : public CudaCommand
{
public:
    CudaEventRecordCommand(CUevent event);
    virtual ~CudaEventRecordCommand();
    virtual void Synchronize() override { CUDA_ASSERT(Driver::EventSynchronize(event_)); }
    virtual bool Synchronizable() override { return true; }
    virtual bool EnableSynchronization() override { return true; }
    // Mark the event_ as destroyed, so that the event_ will be destroyed in the destructor.
    void DestroyEvent() { destroy_event_ = true; }

protected:
    CUevent event_;

private:
    bool destroy_event_ = false;
    virtual CUresult Launch(CUstream stream) override
    { return Driver::EventRecord(event_, stream); }
};

class CudaEventRecordWithFlagsCommand : public CudaEventRecordCommand
{
public:
    CudaEventRecordWithFlagsCommand(CUevent event, unsigned int flags)
        : CudaEventRecordCommand(event), flags_(flags) {}
    virtual ~CudaEventRecordWithFlagsCommand() = default;

private:
    const unsigned int flags_;
    virtual CUresult Launch(CUstream stream) override
    { return Driver::EventRecordWithFlags(event_, stream, flags_); }
};

class CudaEventWaitCommand : public CudaCommand
{
public:
    CudaEventWaitCommand(CUevent event, unsigned int flags)
        : CudaCommand(preempt::kCommandPropertyIdempotent)
        , event_(event), event_cmd_(nullptr), flags_(flags) {}
    CudaEventWaitCommand(std::shared_ptr<CudaEventRecordCommand> event_cmd, unsigned int flags)
        : CudaCommand(preempt::kCommandPropertyIdempotent)
        , event_(nullptr), event_cmd_(event_cmd), flags_(flags) {}
    virtual ~CudaEventWaitCommand() = default;
    virtual void BeforeLaunch() override;

private:
    const CUevent event_; // the event to wait is recorded on a normal cuda stream
    const std::shared_ptr<CudaEventRecordCommand> event_cmd_; // recorded on an XQueue
    const unsigned int flags_;
    virtual CUresult Launch(CUstream stream) override;
};

} // namespace xsched::cuda
