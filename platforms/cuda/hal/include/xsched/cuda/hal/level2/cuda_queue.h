#pragma once

#include "xsched/cuda/hal/level1/cuda_queue.h"
#include "xsched/cuda/hal/level2/mm.h"
#include "xsched/cuda/hal/level2/instrument.h"

namespace xsched::cuda
{

class CudaQueueL2 : public CudaQueueL1
{
public:
    CudaQueueL2(CUstream stream);
    virtual ~CudaQueueL2() = default;

    virtual void Launch(std::shared_ptr<preempt::HwCommand> hw_cmd) override;
    virtual void Deactivate() override;
    virtual void Reactivate(const preempt::CommandLog &log) override;
    virtual void OnPreemptLevelChange(XPreemptLevel level) override;
    virtual void OnHwCommandSubmit(std::shared_ptr<preempt::HwCommand> hw_cmd) override;
    static CUresult DirectLaunch(std::shared_ptr<CudaKernelCommand> kernel, CUstream stream);

    virtual bool          SupportDynamicLevel()  override { return true; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelDeactivate; }

protected:
    std::unique_ptr<InstrumentManager> instrument_manager_ = nullptr;
};

} // namespace xsched::cuda
