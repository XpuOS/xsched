#include "xsched/utils/env.h"
#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/level3/cuda_queue.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

using namespace xsched::cuda;
using namespace xsched::preempt;

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

void CudaQueueLv3Trap::OnHwCommandSubmit(std::shared_ptr<preempt::HwCommand> hw_cmd)
{
    this->CudaQueueLv2::OnHwCommandSubmit(hw_cmd);
    if (level_ < kPreemptLevelInterrupt) return;

    if (std::dynamic_pointer_cast<CudaGraphCommand>(hw_cmd) != nullptr) {
        // do nothing here, will automatically fallback to wait-based preemption
        static bool warned = false;
        if (!warned) {
            warned = true;
            XWARN("CUDA graph cannot support trap-based level-3 preemption, "
                  "falling back to level-1");
        }
        return;
    }
    auto kernel = std::dynamic_pointer_cast<CudaKernelCommand>(hw_cmd);
    // TODO: assign kernel_command->killable
    if (kernel != nullptr) kernel->killable = true;
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
    tsg_context_->Restore();
}

void CudaQueueLv3Tsg::OnPreemptLevelChange(XPreemptLevel level)
{
    XASSERT(level <= kPreemptLevelInterrupt, "unsupported level: %d", level);
    level_ = level;
}
