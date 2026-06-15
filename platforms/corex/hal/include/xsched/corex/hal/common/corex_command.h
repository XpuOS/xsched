#pragma once

#include <memory>
#include <vector>
#include <cstring>

#include "xsched/preempt/hal/hw_command.h"
#include "xsched/corex/hal/common/driver.h"
#include "xsched/corex/hal/common/corex_assert.h"

namespace xsched::corex
{

#define COREXRT_LAUNCH(ptsz, func, ...) \
    (ptsz ? RtDriver::func##_ptsz(__VA_ARGS__) \
          : RtDriver::func(__VA_ARGS__))

class CorexCommand : public preempt::HwCommand
{
public:
    CorexCommand(preempt::XCommandProperties props = preempt::kCommandPropertyNone)
        : HwCommand(props) {}
    virtual ~CorexCommand();

    virtual void Synchronize() override;
    virtual bool Synchronizable() override;
    virtual bool EnableSynchronization() override;
    cudaError_t LaunchWrapper(cudaStream_t stream);

protected:
    virtual cudaError_t Launch(cudaStream_t stream) = 0;

private:
    cudaEvent_t following_event_ = nullptr;
};

class CorexKernelCommand : public CorexCommand
{
public:
    CorexKernelCommand(const void *func, void **params, bool deep_copy);
    virtual ~CorexKernelCommand();

protected:
    const void *func_ = nullptr;
    bool deep_copy_ = false;
    void **params_ = nullptr;

private:
    size_t param_cnt_ = 0;
    char *param_data_ = nullptr;
};

class CorexKernelLaunchCommand : public CorexKernelCommand
{
public:
    CorexKernelLaunchCommand(const void *func, dim3 grid_dim, dim3 block_dim, void **args,
                             size_t shared_mem, bool deep_copy, bool ptsz)
        : CorexKernelCommand(func, args, deep_copy)
        , grid_dim_(grid_dim), block_dim_(block_dim)
        , shared_mem_(shared_mem), ptsz_(ptsz) {}
    virtual ~CorexKernelLaunchCommand() = default;

private:
    virtual cudaError_t Launch(cudaStream_t stream) override
    {
        return COREXRT_LAUNCH(ptsz_, LaunchKernel, func_, grid_dim_, block_dim_, params_, shared_mem_, stream);
    }

    dim3 grid_dim_;
    dim3 block_dim_;
    size_t shared_mem_;
    bool ptsz_;
};

class CorexMemcpyCommand : public CorexCommand
{
public:
    CorexMemcpyCommand(void *dst, const void *src, size_t size, cudaMemcpyKind kind, bool ptsz);
    virtual ~CorexMemcpyCommand();

private:
    virtual cudaError_t Launch(cudaStream_t stream) override
    {
        return COREXRT_LAUNCH(ptsz_, MemcpyAsync, dst_, src_, size_, kind_, stream);
    }

    void *dst_;
    const void *src_;
    size_t size_;
    cudaMemcpyKind kind_;
    bool ptsz_;
    void *pinned_src_ = nullptr;
};

class CorexMemcpy2DCommand : public CorexCommand
{
public:
    CorexMemcpy2DCommand(void *dst, size_t dpitch, const void *src, size_t spitch,
                         size_t width, size_t height, cudaMemcpyKind kind, bool ptsz)
        : dst_(dst), dpitch_(dpitch), src_(src), spitch_(spitch)
        , width_(width), height_(height), kind_(kind), ptsz_(ptsz) {}
    virtual ~CorexMemcpy2DCommand() = default;

private:
    virtual cudaError_t Launch(cudaStream_t stream) override
    {
        return COREXRT_LAUNCH(ptsz_, Memcpy2DAsync, dst_, dpitch_, src_, spitch_, width_, height_, kind_, stream);
    }

    void *dst_;
    size_t dpitch_;
    const void *src_;
    size_t spitch_;
    size_t width_;
    size_t height_;
    cudaMemcpyKind kind_;
    bool ptsz_;
};

class CorexMemcpy3DCommand : public CorexCommand
{
public:
    CorexMemcpy3DCommand(const cudaMemcpy3DParms *p, bool ptsz): params_(*p), ptsz_(ptsz) {}
    virtual ~CorexMemcpy3DCommand() = default;

private:
    virtual cudaError_t Launch(cudaStream_t stream) override
    {
        return COREXRT_LAUNCH(ptsz_, Memcpy3DAsync, &params_, stream);
    }

    cudaMemcpy3DParms params_{};
    bool ptsz_;
};

class CorexMemsetCommand : public CorexCommand
{
public:
    CorexMemsetCommand(void *dev_ptr, int value, size_t count, bool ptsz)
        : dev_ptr_(dev_ptr), value_(value), count_(count), ptsz_(ptsz) {}
    virtual ~CorexMemsetCommand() = default;

private:
    virtual cudaError_t Launch(cudaStream_t stream) override
    {
        return COREXRT_LAUNCH(ptsz_, MemsetAsync, dev_ptr_, value_, count_, stream);
    }

    void *dev_ptr_;
    int value_;
    size_t count_;
    bool ptsz_;
};

class CorexEventRecordCommand : public CorexCommand
{
public:
    CorexEventRecordCommand(cudaEvent_t event, bool ptsz);
    virtual ~CorexEventRecordCommand();

    virtual void Synchronize() override { COREXRT_ASSERT(RtDriver::EventSynchronize(event_)); }
    virtual bool Synchronizable() override { return true; }
    virtual bool EnableSynchronization() override { return true; }
    void DestroyEvent() { destroy_event_ = true; }

protected:
    cudaEvent_t event_ = nullptr;

private:
    virtual cudaError_t Launch(cudaStream_t stream) override
    {
        return COREXRT_LAUNCH(ptsz_, EventRecord, event_, stream);
    }

    bool ptsz_ = false;
    bool destroy_event_ = false;
};

class CorexEventWaitCommand : public CorexCommand
{
public:
    CorexEventWaitCommand(cudaEvent_t event, unsigned int flags, bool ptsz)
        : CorexCommand(preempt::kCommandPropertyIdempotent)
        , event_(event), flags_(flags), ptsz_(ptsz) {}
    CorexEventWaitCommand(std::shared_ptr<CorexEventRecordCommand> event_cmd,
                          unsigned int flags, bool ptsz)
        : CorexCommand(preempt::kCommandPropertyIdempotent)
        , event_cmd_(event_cmd), flags_(flags), ptsz_(ptsz) {}
    virtual ~CorexEventWaitCommand() = default;

    virtual void BeforeLaunch() override;

private:
    virtual cudaError_t Launch(cudaStream_t stream) override;

    cudaEvent_t event_ = nullptr; // the event to wait is recorded on a normal cuda stream
    std::shared_ptr<CorexEventRecordCommand> event_cmd_ = nullptr; // recorded on an XQueue
    unsigned int flags_ = 0;
    bool ptsz_ = false;
};

} // namespace xsched::corex
