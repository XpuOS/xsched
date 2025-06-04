#pragma once

#include <atomic>

#include "xsched/cuda/hal/level2/cuda_queue.h"
#include "xsched/cuda/hal/level3/interrupt.h"
#include "xsched/cuda/hal/level3/tsg.h"

namespace xsched::cuda
{

enum CudaLv3Implementation
{
    kCudaLv3ImplementationTrap,
    kCudaLv3ImplementationTsg,
};

CudaLv3Implementation GetCudaLv3Implementation();

class CudaQueueLv3Trap : public CudaQueueLv2
{
public:
    CudaQueueLv3Trap(CUstream stream);
    virtual ~CudaQueueLv3Trap() = default;

    virtual void Interrupt() override;
    virtual void Restore(const preempt::CommandLog &log) override;
    virtual void OnPreemptLevelChange(XPreemptLevel level) override;
    virtual void OnHwCommandSubmit(std::shared_ptr<preempt::HwCommand> hw_cmd) override;

    virtual bool          SupportDynamicLevel()  override { return true; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelInterrupt; }

protected:
    std::shared_ptr<InterruptContext> interrupt_context_ = nullptr;
};

class CudaQueueLv3Tsg : public CudaQueueLv1
{
public:
    CudaQueueLv3Tsg(CUstream stream);
    virtual ~CudaQueueLv3Tsg() = default;

    virtual void Interrupt() override;
    virtual void Restore(const preempt::CommandLog &log) override;
    virtual void OnPreemptLevelChange(XPreemptLevel level) override;

    virtual bool          SupportDynamicLevel()  override { return true; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelInterrupt; }

private:
    std::shared_ptr<TsgContext> tsg_context_ = nullptr;
};

} // namespace xsched::cuda
