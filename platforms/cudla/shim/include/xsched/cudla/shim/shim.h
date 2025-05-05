#pragma once

#include <memory>

#include "xsched/utils/function.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/cudla/hal/handle.h"
#include "xsched/cudla/hal/driver.h"
#include "xsched/cudla/hal/cudla_command.h"

namespace xsched::cudla
{

inline cudlaStatus XSubmitTask(cudlaDevHandle devHandle, const cudlaTask *ptrToTasks, uint32_t numTasks, void *stream, uint32_t flags)
{
    auto xq = xsched::preempt::HwQueueManager::GetXQueue(GetHwQueueHandle((cudaStream_t)stream));
    if (xq == nullptr) return DlaDriver::SubmitTask(devHandle, ptrToTasks, numTasks, stream, flags);
    auto hw_cmd = std::make_shared<CudlaTaskCommand>(devHandle, ptrToTasks, numTasks, flags);
    xq->Submit(hw_cmd);
    return cudlaSuccess;
}

inline cudaError_t XMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)
{
    auto xq = xsched::preempt::HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return RtDriver::MemcpyAsync(dst, src, count, kind, stream);
    auto hw_cmd = std::make_shared<CudlaMemoryCommand>(dst, src, count, kind);
    xq->Submit(hw_cmd);
    return cudaSuccess;
}

cudaError_t XStreamCreate(cudaStream_t *stream);
cudaError_t XStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags);
cudaError_t XStreamSynchronize(cudaStream_t stream);

cudaError_t XEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t XEventDestroy(cudaEvent_t event);
cudaError_t XEventSynchronize(cudaEvent_t event);

} // namespace xsched::cudla
