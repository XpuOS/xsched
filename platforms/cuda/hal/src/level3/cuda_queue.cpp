#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/level2/op_stream.h"
#include "xsched/cuda/hal/level3/cuda_queue.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

using namespace xsched::cuda;
using namespace xsched::preempt;

CudaQueueL3::CudaQueueL3(CUstream stream): CudaQueueL2(stream)
    , kOpStream(OpStreamManager::GetOpStream(context_))
{
    trap_manager_ = TrapManager::GetTrapManager(context_);
}

void CudaQueueL3::Interrupt()
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Interrupt() not supported on level-%d", level_);
    
    // FIXME: what if multiple threads call Interrupt()?
    CUDA_ASSERT(Driver::StreamSynchronize(kOpStream));
    trap_manager_->InterruptContext();
}

void CudaQueueL3::Restore(const CommandLog &)
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Restore() not supported on level-%d", level_);
}

void CudaQueueL3::OnPreemptLevelChange(XPreemptLevel level)
{
    XASSERT(level <= kPreemptLevelInterrupt, "unsupported level: %d", level);
    if (level == kPreemptLevelInterrupt) trap_manager_->SetTrapHandler();
    level_ = level;
}

void CudaQueueL3::OnHwCommandSubmit(std::shared_ptr<preempt::HwCommand> cmd)
{
    if (level_ < kPreemptLevelDeactivate) return;
    auto kernel = std::dynamic_pointer_cast<CudaKernelCommand>(cmd);
    if (kernel == nullptr) return;
    instrument_manager_->Instrument(kernel);
    // TODO: assign kernel_command->killable
    kernel->killable = true;
}

CUresult CudaQueueL3::DirectLaunch(std::shared_ptr<CudaKernelCommand> kernel, CUstream stream)
{
    CUcontext context;
    CUDA_ASSERT(Driver::StreamGetCtx(stream, &context));
    auto ctx = InstrumentContext::GetInstrumentContext(context);
    return ctx->Launch(kernel, stream, kKernelLaunchDefault);
}
