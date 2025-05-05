#include "xsched/xqueue.h"
#include "xsched/utils/map.h"
#include "xsched/protocol/def.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/vpi/hal.h"
#include "xsched/vpi/shim/shim.h"
#include "xsched/vpi/hal/vpi_queue.h"
#include "xsched/vpi/hal/vpi_command.h"

using namespace xsched::preempt;

namespace xsched::vpi
{

static utils::ObjectMap<VPIEvent, std::shared_ptr<VpiEventRecordCommand>> g_events;

VPIStatus XEventRecord(VPIEvent event, VPIStream stream)
{
    XDEBG("XEventRecord(event: %p, stream: %p)", event, stream);
    if (event == nullptr || stream == nullptr) return Driver::EventRecord(event, stream);

    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return Driver::EventRecord(event, stream);

    auto xevent = std::make_shared<VpiEventRecordCommand>(event);
    xq->Submit(xevent);
    g_events.Add(event, xevent);
    return VPI_SUCCESS;
}

VPIStatus XEventSync(VPIEvent event)
{
    XDEBG("XEventSync(event: %p)", event);
    auto xevent = g_events.Get(event, nullptr);
    if (xevent == nullptr) return Driver::EventSync(event);
    xevent->Wait();
    return VPI_SUCCESS;
}

void XEventDestroy(VPIEvent event)
{
    XDEBG("XEventDestroy(event: %p)", event);
    if (event == nullptr) return Driver::EventDestroy(event);
    g_events.Del(event, nullptr);
    Driver::EventDestroy(event);
}

VPIStatus XStreamSync(VPIStream stream)
{
    XDEBG("XStreamSync(stream: %p)", stream);
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return Driver::StreamSync(stream);
    xq->WaitAll();
    return VPI_SUCCESS;
}

VPIStatus XStreamCreate(uint64_t flags, VPIStream *stream)
{
    auto res = Driver::StreamCreate(flags, stream);
    if (res != VPI_SUCCESS) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return VpiQueueCreate(hwq, *stream); });
    XDEBG("XStreamCreate(stream: %p)", *stream);
    return res;
}

} // namespace xsched::vpi
