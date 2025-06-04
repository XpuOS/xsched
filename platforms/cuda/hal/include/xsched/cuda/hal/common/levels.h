#pragma once

#include "xsched/types.h"
#include "xsched/utils/log.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/common/cuda_command.h"

namespace xsched::cuda
{

std::shared_ptr<preempt::HwQueue> CudaQueueCreate(CUstream stream);
CUresult DirectLaunch(std::shared_ptr<CudaKernelCommand> kernel, CUstream stream);

} // namespace xsched::cuda
