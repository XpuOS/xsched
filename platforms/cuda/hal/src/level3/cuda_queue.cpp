#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/level3/cuda_queue.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

using namespace xsched::cuda;
using namespace xsched::preempt;

CudaLv3Implementation xsched::cuda::GetCudaLv3Implementation()
{
    static CudaLv3Implementation impl = []()->CudaLv3Implementation {
        char *str = std::getenv(XSCHED_CUDA_LV3_IMPL_ENV_NAME);
        if (str == nullptr) return kCudaLv3ImplementationTrap;
        if (strcmp(str, "TSG") == 0) return kCudaLv3ImplementationTsg;
        return kCudaLv3ImplementationTrap;
    }();
    return impl;
}

CudaQueueLv3Trap::CudaQueueLv3Trap(CUstream stream): CudaQueueLv2(stream)
{
    interrupt_context_ = InterruptContext::Instance(context_);
}

void CudaQueueLv3Trap::Interrupt()
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Interrupt() not supported on level-%d", level_);
    // wait until the preempt flag is set
    CUDA_ASSERT(Driver::StreamSynchronize(instrument_manager_->OpStream()));
    // FIXME: what if multiple threads call Interrupt()?
    interrupt_context_->Interrupt();
}

void CudaQueueLv3Trap::Restore(const CommandLog &)
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Restore() not supported on level-%d", level_);
}

void CudaQueueLv3Trap::OnPreemptLevelChange(XPreemptLevel level)
{
    XASSERT(level <= kPreemptLevelInterrupt, "unsupported level: %d", level);
    if (level == kPreemptLevelInterrupt) {
        instrument_manager_->NotifyTrapInstrumented();
        interrupt_context_->InstrumentTrapHandler();
    }
    level_ = level;
}

void CudaQueueLv3Trap::OnHwCommandSubmit(std::shared_ptr<preempt::HwCommand> cmd)
{
    if (level_ < kPreemptLevelDeactivate) return;
    auto kernel = std::dynamic_pointer_cast<CudaKernelCommand>(cmd);
    if (kernel == nullptr) return;
    instrument_manager_->Instrument(kernel);
    // TODO: assign kernel_command->killable
    kernel->killable = true;
}

CudaQueueLv3Tsg::CudaQueueLv3Tsg(CUstream stream): CudaQueueLv1(stream)
{
    tsg_context_ = TsgContext::Instance(context_);
}

void CudaQueueLv3Tsg::Interrupt()
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Interrupt() not supported on level-%d", level_);
    tsg_context_->Interrupt();
}

void CudaQueueLv3Tsg::Restore(const CommandLog &)
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Restore() not supported on level-%d", level_);
}

void CudaQueueLv3Tsg::OnPreemptLevelChange(XPreemptLevel level)
{
    XASSERT(level <= kPreemptLevelInterrupt, "unsupported level: %d", level);
    level_ = level;
}
