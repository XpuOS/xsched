#pragma once

#include <chrono>
#include <memory>
#include <cstring>
#include <cstddef>

#include "xsched/types.h"
#include "xsched/utils/common.h"
#include "xsched/sched/protocol/hint.h"
#include "xsched/sched/protocol/status.h"

namespace xsched::sched
{

enum EventType
{
    kEventHint               = 0,
    kEventSchedulerTerminate = 1,
    kEventProcessCreate      = 2,
    kEventProcessDestroy     = 3,
    kEventXQueueCreate       = 4,
    kEventXQueueDestroy      = 5,
    kEventXQueueReady        = 6,
    kEventXQueueIdle         = 7,
    kEventXQueueConfigUpdate = 8,
    kEventXQueueQuery        = 9,
    kEventXQueueQueryAll     = 10,
};

struct EventMeta
{
    EventType type;
    PID       pid;
};

class Event
{
public:
    Event() = default;
    virtual ~Event() = default;

    /// @brief Get the data of the event. MUST start with EventMeta.
    virtual const void *Data() const = 0;
    virtual size_t      Size() const = 0;
    virtual EventType   Type() const = 0;
    virtual PID         Pid()  const = 0;

    static std::shared_ptr<const Event> CopyConstructor(const void *data);
};

class HintEvent : public Event
{
public:
    HintEvent(const void *data);
    HintEvent(std::shared_ptr<const Hint> hint);
    virtual ~HintEvent();

    virtual const void *Data() const override { return data_; }
    virtual size_t      Size() const override { return data_->size; }
    virtual EventType   Type() const override { return kEventHint; }
    virtual PID         Pid()  const override { return data_->meta.pid; }
    std::shared_ptr<const Hint> GetHint() const
    { return Hint::CopyConstructor(data_->hint_data); }

private:
    struct EventData
    {
        EventMeta meta;
        /// @brief The size of the whole EventData.
        size_t    size;
        /// @brief 64 is only a placeholder.
        /// The actual size of the hint data can be calculated by
        /// size - offsetof(EventData, hint_data).
        char      hint_data[64];
    };

    const EventData *data_ = nullptr;
};

/// @brief Terminate the worker thread of the scheduler.
class SchedulerTerminateEvent : public Event
{
public:
    SchedulerTerminateEvent(): meta_{ .type = kEventSchedulerTerminate, .pid = GetProcessId() } {}
    SchedulerTerminateEvent(const void *data): meta_(*(const EventMeta *)data) {}
    virtual ~SchedulerTerminateEvent() = default;

    virtual const void *Data() const override { return (void *)&meta_; }
    virtual size_t      Size() const override { return sizeof(meta_); }
    virtual EventType   Type() const override { return kEventSchedulerTerminate; }
    virtual PID         Pid()  const override { return meta_.pid; }

private:
    EventMeta meta_;
};

class ProcessCreateEvent : public Event
{
public:
    ProcessCreateEvent(const std::string &cmdline);
    ProcessCreateEvent(const void *data): data_(*(const EventData *)data) {}
    virtual ~ProcessCreateEvent() = default;

    virtual const void *Data() const override { return (void *)&data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual EventType   Type() const override { return kEventProcessCreate; }
    virtual PID         Pid()  const override { return data_.meta.pid; }

    const char *Cmdline() const { return data_.cmdline; }
    static size_t CmdlineCapacity() { return sizeof(data_.cmdline); }

private:
    struct EventData
    {
        EventMeta meta;
        char      cmdline[256];
    };

    EventData data_;
};

class ProcessDestroyEvent : public Event
{
public:
    ProcessDestroyEvent(): meta_{ .type = kEventProcessDestroy, .pid = GetProcessId() } {}
    ProcessDestroyEvent(PID pid): meta_{ .type = kEventProcessDestroy, .pid = pid } {}
    ProcessDestroyEvent(const void *data): meta_(*(const EventMeta *)data) {}
    virtual ~ProcessDestroyEvent() = default;

    virtual const void *Data() const override { return (void *)&meta_; }
    virtual size_t      Size() const override { return sizeof(meta_); }
    virtual EventType   Type() const override { return kEventProcessDestroy; }
    virtual PID         Pid()  const override { return meta_.pid; }

private:
    EventMeta meta_;
};

class XQueueCreateEvent : public Event
{
public:
    XQueueCreateEvent(XQueueHandle handle, XDevice device, XPreemptLevel level,
                      int64_t threshold, int64_t batch_size)
        : data_{
            .meta = {
                .type = kEventXQueueCreate,
                .pid = GetProcessId()
            },
            .handle = handle,
            .device = device,
            .level = level,
            .threshold = threshold,
            .batch_size = batch_size
        } {}
    XQueueCreateEvent(const void *data): data_(*(const EventData *)data) {}
    virtual ~XQueueCreateEvent() = default;

    virtual const void *Data() const override { return (void *)&data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual EventType   Type() const override { return kEventXQueueCreate; }
    virtual PID         Pid()  const override { return data_.meta.pid; }

    XQueueHandle  Handle()    const { return data_.handle; }
    XDevice       Device()    const { return data_.device; }
    XPreemptLevel Level()     const { return data_.level; }
    int64_t       Threshold() const { return data_.threshold; }
    int64_t       BatchSize() const { return data_.batch_size; }

private:
    struct EventData
    {
        EventMeta     meta;
        XQueueHandle  handle;
        XDevice       device;
        XPreemptLevel level;
        int64_t       threshold;
        int64_t       batch_size;
    };

    EventData data_;
};

class XQueueDestroyEvent : public Event
{
public:
    XQueueDestroyEvent(XQueueHandle handle)
        : data_{
            .meta = {
                .type = kEventXQueueDestroy,
                .pid = GetProcessId()
            },
            .handle = handle
        } {}
    XQueueDestroyEvent(const void *data): data_(*(const EventData *)data) {}
    virtual ~XQueueDestroyEvent() = default;

    virtual const void *Data() const override { return (void *)&data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual EventType   Type() const override { return kEventXQueueDestroy; }
    virtual PID         Pid()  const override { return data_.meta.pid; }

    XQueueHandle Handle() const { return data_.handle; }

private:
    struct EventData
    {
        EventMeta    meta;
        XQueueHandle handle;
    };

    EventData data_;
};

class XQueueReadyEvent : public Event
{
public:
    XQueueReadyEvent(XQueueHandle handle, std::chrono::system_clock::time_point ready_time)
        : data_{
            .meta = {
                .type = kEventXQueueReady,
                .pid = GetProcessId()
            },
            .handle = handle,
            .ready_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                ready_time.time_since_epoch()).count()
        } {}
    XQueueReadyEvent(const void *data): data_(*(const EventData *)data) {}
    virtual ~XQueueReadyEvent() = default;

    virtual const void *Data() const override { return (void *)&data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual EventType   Type() const override { return kEventXQueueReady; }
    virtual PID         Pid()  const override { return data_.meta.pid; }

    XQueueHandle Handle() const { return data_.handle; }
    std::chrono::system_clock::time_point ReadyTime() const
    {
        return std::chrono::system_clock::time_point(
            std::chrono::microseconds(data_.ready_time_us));
    }

private:
    struct EventData
    {
        EventMeta    meta;
        XQueueHandle handle;
        int64_t      ready_time_us;
    };

    EventData data_;
};

class XQueueIdleEvent : public Event
{
public:
    XQueueIdleEvent(XQueueHandle handle)
        : data_{
            .meta = {
                .type = kEventXQueueIdle,
                .pid = GetProcessId()
            },
            .handle = handle
        } {}
    XQueueIdleEvent(const void *data): data_(*(const EventData *)data) {}
    virtual ~XQueueIdleEvent() = default;

    virtual const void *Data() const override { return (void *)&data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual EventType   Type() const override { return kEventXQueueIdle; }
    virtual PID         Pid()  const override { return data_.meta.pid; }

    XQueueHandle Handle() const { return data_.handle; }

private:
    struct EventData
    {
        EventMeta    meta;
        XQueueHandle handle;
    };

    EventData data_;
};

class XQueueConfigUpdateEvent : public Event
{
public:
    XQueueConfigUpdateEvent(XQueueHandle handle, XDevice device, XPreemptLevel level,
                            int64_t threshold, int64_t batch_size)
        : data_{
            .meta = {
                .type = kEventXQueueConfigUpdate,
                .pid = GetProcessId()
            },
            .handle = handle,
            .device = device,
            .level = level,
            .threshold = threshold,
            .batch_size = batch_size
        } {}
    XQueueConfigUpdateEvent(const void *data): data_(*(const EventData *)data) {}
    virtual ~XQueueConfigUpdateEvent() = default;

    virtual const void *Data() const override { return (void *)&data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual EventType   Type() const override { return kEventXQueueConfigUpdate; }
    virtual PID         Pid()  const override { return data_.meta.pid; }

    XQueueHandle  Handle()    const { return data_.handle; }
    XDevice       Device()    const { return data_.device; }
    XPreemptLevel Level()     const { return data_.level; }
    int64_t       Threshold() const { return data_.threshold; }
    int64_t       BatchSize() const { return data_.batch_size; }

private:
    struct EventData
    {
        EventMeta     meta;
        XQueueHandle  handle;
        XDevice       device;
        XPreemptLevel level;
        int64_t       threshold;
        int64_t       batch_size;
    };

    EventData data_;
};

/// @brief Query the status of a given XQueue.
/// @note The event can only be sent by the server to the scheduler in the same process.
class XQueueQueryEvent : public Event
{
public:
    XQueueQueryEvent(XQueueHandle handle, StatusQuery *query_data)
        : data_{
            .meta = {
                .type = kEventXQueueQuery,
                .pid = GetProcessId()
            },
            .handle = handle,
            .query_data = query_data
        } {}
    XQueueQueryEvent(const void *data): data_(*(const EventData *)data) {}
    virtual ~XQueueQueryEvent() = default;

    virtual const void *Data() const override { return (void *)&data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual EventType   Type() const override { return kEventXQueueQuery; }
    virtual PID         Pid()  const override { return data_.meta.pid; }

    XQueueHandle Handle()    const { return data_.handle; }
    StatusQuery *QueryData() const;

private:
    struct EventData
    {
        EventMeta    meta;
        XQueueHandle handle;
        StatusQuery *query_data;
    };

    EventData data_;
};

/// @brief Query the status of all XQueues.
/// @note The event can only be sent by the server to the scheduler in the same process.
class XQueueQueryAllEvent : public Event
{
public:
    XQueueQueryAllEvent(StatusQuery *query_data)
        : data_{
            .meta = {
                .type = kEventXQueueQueryAll,
                .pid = GetProcessId()
            },
            .query_data = query_data
        } {}
    XQueueQueryAllEvent(const void *data): data_(*(const EventData *)data) {}
    
    virtual ~XQueueQueryAllEvent() = default;

    virtual const void *Data() const override { return (void *)&data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual EventType   Type() const override { return kEventXQueueQueryAll; }
    virtual PID         Pid()  const override { return data_.meta.pid; }

    StatusQuery *QueryData() const;

private:
    struct EventData
    {
        EventMeta    meta;
        StatusQuery *query_data;
    };

    EventData data_;
};

} // namespace xsched::sched
