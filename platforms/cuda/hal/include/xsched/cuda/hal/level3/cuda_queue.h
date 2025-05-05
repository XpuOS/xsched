#pragma once

#include "xsched/cuda/hal/level2/cuda_queue.h"
#include "xsched/cuda/hal/level3/trap.h"

namespace xsched::cuda
{

class CudaQueueL3 : public CudaQueueL2
{
public:
    CudaQueueL3(CUstream stream);
    virtual ~CudaQueueL3() = default;

    virtual void Interrupt() override;
    virtual void Restore(const preempt::CommandLog &log) override;
    virtual void OnPreemptLevelChange(XPreemptLevel level) override;
    virtual void OnHwCommandSubmit(std::shared_ptr<preempt::HwCommand> hw_cmd) override;
    static CUresult DirectLaunch(std::shared_ptr<CudaKernelCommand> kernel, CUstream stream);

    virtual bool          SupportDynamicLevel()  override { return true; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelInterrupt; }

protected:
    const CUstream kOpStream;
    std::shared_ptr<TrapManager> trap_manager_ = nullptr;
};

} // namespace xsched::cuda
