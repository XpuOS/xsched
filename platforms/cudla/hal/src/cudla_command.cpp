#include <cstring>

#include "xsched/utils/xassert.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/cudla/hal/event_pool.h"
#include "xsched/cudla/hal/cudla_command.h"

using namespace xsched::cudla;
using namespace xsched::preempt;

CudlaCommand::~CudlaCommand()
{
    if (following_event_ == nullptr) return;
    EventPool::Instance().Push(following_event_);
}

void CudlaCommand::Synchronize()
{
    XASSERT(following_event_ != nullptr,
            "following_event_ is nullptr, EnableSynchronization() should be called first");
    CUDART_ASSERT(RtDriver::EventSynchronize(following_event_));
}

bool CudlaCommand::Synchronizable()
{
    return following_event_ != nullptr;
}

bool CudlaCommand::EnableSynchronization()
{
    following_event_ = (cudaEvent_t)EventPool::Instance().Pop();
    return following_event_ != nullptr;
}

cudaError_t CudlaCommand::LaunchWrapper(cudaStream_t stream)
{
    cudaError_t ret = Launch(stream);
    if (UNLIKELY(ret != cudaSuccess)) return ret;
    if (following_event_ != nullptr) ret = RtDriver::EventRecord(following_event_, stream);
    return ret;
}

CudlaTaskCommand::CudlaTaskCommand(cudlaDevHandle const dev_handle, const cudlaTask * const tasks,
                                   uint32_t const num_tasks, uint32_t const flags)
    : CudlaCommand(kCommandPropertyNone)
    , dev_handle_(dev_handle), num_tasks_(num_tasks), flags_(flags)
{
    XASSERT(tasks != nullptr, "tasks should not be nullptr");
    tasks_ = (cudlaTask *)malloc(sizeof(cudlaTask) * num_tasks);
    memcpy(tasks_, tasks, sizeof(cudlaTask) * num_tasks);
}

CudlaTaskCommand::~CudlaTaskCommand()
{
    if (tasks_ != nullptr) free(tasks_);
}

CudlaEventRecordCommand::CudlaEventRecordCommand(cudaEvent_t event)
    : CudlaCommand(kCommandPropertyIdempotent), event_(event)
{
    XASSERT(event_ != nullptr, "cuda event should not be nullptr");
}
