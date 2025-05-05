#pragma once

#include "xsched/cudla/hal/driver.h"
#include "xsched/cudla/hal/cudla_assert.h"
#include "xsched/preempt/hal/hw_command.h"

namespace xsched::cudla
{

class CudlaCommand : public preempt::HwCommand
{
public:
    CudlaCommand(preempt::XCommandProperties props): HwCommand(props) {}
    virtual ~CudlaCommand();
    virtual void Synchronize() override;
    virtual bool Synchronizable() override;
    virtual bool EnableSynchronization() override;
    cudaError_t LaunchWrapper(cudaStream_t stream);

private:
    virtual cudaError_t Launch(cudaStream_t stream) = 0;
    cudaEvent_t following_event_ = nullptr;
};

class CudlaTaskCommand : public CudlaCommand
{
public:
    CudlaTaskCommand(cudlaDevHandle const dev_handle, const cudlaTask * const tasks,
                     uint32_t const num_tasks, uint32_t const flags);
    virtual ~CudlaTaskCommand();

private:
    cudlaDevHandle const dev_handle_;
    cudlaTask *tasks_ = nullptr;
    uint32_t const num_tasks_;
    uint32_t const flags_;
    virtual cudaError_t Launch(cudaStream_t stream) override
    {
        CUDLA_ASSERT(DlaDriver::SubmitTask(dev_handle_, tasks_, num_tasks_, stream, flags_));
        return cudaSuccess;
    }
};

class CudlaMemoryCommand : public CudlaCommand
{
public:
    CudlaMemoryCommand(void *dst, const void *src, size_t size, cudaMemcpyKind kind)
        : CudlaCommand(preempt::kCommandPropertyNone)
        , dst_(dst), src_(src), size_(size), kind_(kind) {}
    virtual ~CudlaMemoryCommand() = default;

private:
    void *dst_;
    const void *src_;
    size_t size_;
    cudaMemcpyKind kind_;
    virtual cudaError_t Launch(cudaStream_t stream) override
    { return RtDriver::MemcpyAsync(dst_, src_, size_, kind_, stream); }
};

class CudlaEventRecordCommand : public CudlaCommand
{
public:
    CudlaEventRecordCommand(cudaEvent_t event);
    virtual ~CudlaEventRecordCommand() = default;
    virtual void Synchronize() override { CUDART_ASSERT(RtDriver::EventSynchronize(event_)); }
    virtual bool Synchronizable() override { return true; }
    virtual bool EnableSynchronization() override { return true; }

private:
    cudaEvent_t event_;
    virtual cudaError_t Launch(cudaStream_t stream) override
    { return RtDriver::EventRecord(event_, stream); }
};

} // namespace xsched::cudla
