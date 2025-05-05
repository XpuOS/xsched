#include <cstring>

#include "xsched/utils/xassert.h"
#include "xsched/sched/protocol/event.h"

using namespace xsched::sched;

std::shared_ptr<const Event> Event::CopyConstructor(const void *data)
{
    auto meta = (const EventMeta *)data;
    switch (meta->type)
    {
    case kEventHint:
        return std::make_shared<HintEvent>(data);
    case kEventSchedulerTerminate:
        return std::make_shared<SchedulerTerminateEvent>(data);
    case kEventProcessCreate:
        return std::make_shared<ProcessCreateEvent>(data);
    case kEventProcessDestroy:
        return std::make_shared<ProcessDestroyEvent>(data);
    case kEventXQueueCreate:
        return std::make_shared<XQueueCreateEvent>(data);
    case kEventXQueueDestroy:
        return std::make_shared<XQueueDestroyEvent>(data);
    case kEventXQueueReady:
        return std::make_shared<XQueueReadyEvent>(data);
    case kEventXQueueIdle:
        return std::make_shared<XQueueIdleEvent>(data);
    case kEventXQueueConfigUpdate:
        return std::make_shared<XQueueConfigUpdateEvent>(data); 
    case kEventXQueueQuery:
        return std::make_shared<XQueueQueryEvent>(data);
    case kEventXQueueQueryAll:
        return std::make_shared<XQueueQueryAllEvent>(data);
    default:
        XASSERT(false, "unknown event type: %d", meta->type);
        return nullptr;
    }
}

HintEvent::HintEvent(const void *data)
{
    size_t size = ((const EventData *)data)->size;
    EventData *new_data = (EventData *)malloc(size);
    memcpy(new_data, data, size);
    data_ = new_data;
}

HintEvent::HintEvent(std::shared_ptr<const Hint> hint)
{
    size_t size = offsetof(EventData, hint_data) + hint->Size();
    EventData *data = (EventData *)malloc(size);

    data->meta.type = kEventHint;
    data->meta.pid = GetProcessId();
    data->size = size;
    memcpy(data->hint_data, hint->Data(), hint->Size());

    data_ = data;
}

HintEvent::~HintEvent()
{
    if (data_) free((void *)data_);
}

ProcessCreateEvent::ProcessCreateEvent(const std::string &cmdline)
    : data_{
        .meta = {
            .type = kEventProcessCreate,
            .pid = GetProcessId()
        },
        .cmdline = {0}
    }
{
    strncpy(data_.cmdline, cmdline.c_str(), sizeof(data_.cmdline));
    data_.cmdline[sizeof(data_.cmdline) - 1] = '\0';
}

StatusQuery *XQueueQueryEvent::QueryData() const
{
    XASSERT(data_.query_data != nullptr, "query data should not be nullptr");
    XASSERT(data_.meta.pid == GetProcessId(),
            "query data should only be accessed by the process creating it");
    return data_.query_data;
}

StatusQuery *XQueueQueryAllEvent::QueryData() const
{
    XASSERT(data_.query_data != nullptr, "query data should not be nullptr");
    XASSERT(data_.meta.pid == GetProcessId(),
            "query data should only be accessed by the process creating it");
    return data_.query_data;
}
