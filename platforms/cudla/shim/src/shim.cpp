#include "xsched/xqueue.h"
#include "xsched/utils/map.h"
#include "xsched/protocol/def.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/cudla/hal.h"
#include "xsched/cudla/shim/shim.h"
#include "xsched/cudla/hal/cudla_queue.h"
#include "xsched/cudla/hal/cudla_command.h"

using namespace xsched::preempt;

namespace xsched::cudla
{

static utils::ObjectMap<cudaEvent_t, std::shared_ptr<CudlaEventRecordCommand>> g_events;

cudaError_t XStreamCreate(cudaStream_t *stream)
{
    auto res = RtDriver::StreamCreate(stream);
    if (res != cudaSuccess) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return CudlaQueueCreate(hwq, *stream); });
    XDEBG("XStreamCreate(stream: %p)", *stream);
    return res;
}

cudaError_t XStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags)
{
    auto res = RtDriver::StreamCreateWithFlags(stream, flags);
    if (res != cudaSuccess) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return CudlaQueueCreate(hwq, *stream); });
    XDEBG("XStreamCreateWithFlags(stream: %p, flags: %lx)", *stream, flags);
    return res;
}

cudaError_t XStreamSynchronize(cudaStream_t stream)
{
    XDEBG("XStreamSynchronize(stream: %p)", stream);
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return RtDriver::StreamSynchronize(stream);
    xq->WaitAll();
    return cudaSuccess;
}

cudaError_t XEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    XDEBG("XEventRecord(event: %p, stream: %p)", event, stream);
    if (event == nullptr || stream == nullptr) return RtDriver::EventRecord(event, stream);

    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return RtDriver::EventRecord(event, stream);

    auto xevent = std::make_shared<CudlaEventRecordCommand>(event);
    xq->Submit(xevent);
    g_events.Add(event, xevent);
    return cudaSuccess;
}

cudaError_t XEventSynchronize(cudaEvent_t event)
{
    XDEBG("XEventSynchronize(event: %p)", event);
    auto xevent = g_events.Get(event, nullptr);
    if (xevent == nullptr) return RtDriver::EventSynchronize(event);
    xevent->Wait();
    return cudaSuccess;
}

cudaError_t XEventDestroy(cudaEvent_t event)
{
    XDEBG("XEventDestroy(event: %p)", event);
    if (event == nullptr) return RtDriver::EventDestroy(event);
    g_events.Del(event, nullptr);
    return RtDriver::EventDestroy(event);
}

} // namespace xsched::cudla
