#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/level2/cuda_queue.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

using namespace xsched::cuda;
using namespace xsched::preempt;

CudaQueueL2::CudaQueueL2(CUstream stream): CudaQueueL1(stream)
{
    instrument_manager_ = std::make_unique<InstrumentManager>(context_, stream);
}

void CudaQueueL2::Launch(std::shared_ptr<HwCommand> hw_cmd)
{
    auto kernel = std::dynamic_pointer_cast<CudaKernelCommand>(hw_cmd);
    if (kernel != nullptr) return instrument_manager_->Launch(kernel, level_);
    
    auto cuda_cmd = std::dynamic_pointer_cast<CudaCommand>(hw_cmd);
    XASSERT(cuda_cmd != nullptr, "hw_cmd is not a CudaCommand");
    CUDA_ASSERT(cuda_cmd->LaunchWrapper(kStream));
}

void CudaQueueL2::Deactivate()
{
    XASSERT(level_ >= kPreemptLevelDeactivate, "Deactivate() not supported on level-%d", level_);
    instrument_manager_->Deactivate();
}

void CudaQueueL2::Reactivate(const preempt::CommandLog &log)
{
    XASSERT(level_ >= kPreemptLevelDeactivate, "Reactivate() not supported on level-%d", level_);
    this->Synchronize();

    uint64_t resume_cmd_idx = instrument_manager_->Reactivate();
    if (resume_cmd_idx == 0) return;

    for (auto cmd : log) {
        if (cmd->GetIdx() < (int64_t)resume_cmd_idx) continue;
        this->Launch(cmd);
    }
}

void CudaQueueL2::OnPreemptLevelChange(XPreemptLevel level)
{
    XASSERT(level <= kPreemptLevelDeactivate, "unsupported level: %d", level);
    level_ = level;
}

void CudaQueueL2::OnHwCommandSubmit(std::shared_ptr<preempt::HwCommand> hw_cmd)
{
    if (level_ < kPreemptLevelDeactivate) return;
    auto kernel = std::dynamic_pointer_cast<CudaKernelCommand>(hw_cmd);
    if (kernel != nullptr) instrument_manager_->Instrument(kernel);
}

CUresult CudaQueueL2::DirectLaunch(std::shared_ptr<CudaKernelCommand> kernel,
                                   CUstream stream)
{
    CUcontext context;
    CUDA_ASSERT(Driver::StreamGetCtx(stream, &context));
    auto ctx = InstrumentContext::GetInstrumentContext(context);
    return ctx->Launch(kernel, stream, kKernelLaunchDefault);
}
