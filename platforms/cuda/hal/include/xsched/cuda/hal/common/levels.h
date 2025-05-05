#pragma once

#include "xsched/types.h"
#include "xsched/utils/log.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/common/cuda_command.h"

namespace xsched::cuda
{

std::shared_ptr<preempt::HwQueue> MakeCudaQueue(CUstream stream);
CUresult DirectLaunch(std::shared_ptr<CudaKernelCommand> kernel, CUstream stream);

} // namespace xsched::cuda

#define SET_CUDA_QUEUE_LEVEL(level) \
namespace xsched::cuda \
{ \
std::shared_ptr<preempt::HwQueue> MakeCudaQueue(CUstream stream)  \
{ \
    return std::make_shared<CudaQueueL##level>(stream); \
} \
CUresult DirectLaunch(std::shared_ptr<CudaKernelCommand> kernel, CUstream stream) \
{ \
    return CudaQueueL##level::DirectLaunch(kernel, stream); \
} \
} // namespace xsched::cuda
