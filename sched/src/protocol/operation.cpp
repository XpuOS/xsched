#include <cstring>

#include "xsched/utils/log.h"
#include "xsched/sched/protocol/operation.h"

using namespace xsched::sched;

std::shared_ptr<const Operation> Operation::CopyConstructor(const void *data)
{
    auto meta = (const OperationMeta *)data;
    switch (meta->type)
    {
    case kOperationTerminate:
        return std::make_shared<TerminateOperation>(data);
    case kOperationSched:
        return std::make_shared<SchedOperation>(data);
    case kOperationConfig:
        return std::make_shared<ConfigOperation>(data);
    default:
        XERRO("unsupported operation type: %d", meta->type);
        return nullptr;
    }
}

SchedOperation::SchedOperation(const void *data)
{
    size_t size = ((const OperationData *)data)->size;
    OperationData *new_data = (OperationData *)malloc(size);
    memcpy(new_data, data, size);
    data_ = new_data;
}

SchedOperation::SchedOperation(OperationId id, const ProcessStatus &status)
{
    PID pid = status.info.pid;
    size_t running_cnt = status.running_xqueues.size();
    size_t suspended_cnt = status.suspended_xqueues.size();
    size_t xqueue_cnt = running_cnt + suspended_cnt;
    
    size_t size = offsetof(OperationData, handles) + xqueue_cnt * sizeof(XQueueHandle);
    OperationData *data = (OperationData *)malloc(size);
    data_ = data;

    data->meta.id = id;
    data->meta.type = kOperationSched;
    data->meta.pid = pid;
    data->size = size;
    data->running_cnt = running_cnt;
    data->suspended_cnt = suspended_cnt;

    xqueue_cnt = 0;
    for (auto &handle : status.running_xqueues) data->handles[xqueue_cnt++] = handle;
    for (auto &handle : status.suspended_xqueues) data->handles[xqueue_cnt++] = handle;
}

SchedOperation::SchedOperation(OperationId id, PID pid, SchedType type,
                               const std::vector<XQueueHandle> &xqs)
{
    size_t xqueue_cnt = xqs.size();
    size_t running_cnt = type == kSchedTypeResumeAll ? xqueue_cnt : 0;
    size_t suspended_cnt = type == kSchedTypeSuspendAll ? xqueue_cnt : 0;

    size_t size = offsetof(OperationData, handles) + xqueue_cnt * sizeof(XQueueHandle);
    OperationData *data = (OperationData *)malloc(size);
    data_ = data;

    data->meta.id = id;
    data->meta.type = kOperationSched;
    data->meta.pid = pid;
    data->size = size;
    data->running_cnt = running_cnt;
    data->suspended_cnt = suspended_cnt;

    memcpy(data->handles, xqs.data(), xqueue_cnt * sizeof(XQueueHandle));
}

SchedOperation::~SchedOperation()
{
    if (data_) free((void *)data_);
}

ConfigOperation::ConfigOperation(const void *data)
{
    size_t size = ConfigOperation::Size();
    OperationData *new_data = (OperationData *)malloc(size);
    memcpy(new_data, data, size);
    data_ = new_data;
}

ConfigOperation::ConfigOperation(OperationId id, PID pid, XQueueHandle handle,
                                 XPreemptLevel level, int64_t threshold, int64_t batch_size)
{
    OperationData *data = (OperationData *)malloc(ConfigOperation::Size());

    data->meta.id = id;
    data->meta.type = kOperationConfig;
    data->meta.pid = pid;
    data->handle = handle;
    data->level = level;
    data->threshold = threshold;
    data->batch_size = batch_size;

    data_ = data;
}

ConfigOperation::~ConfigOperation()
{
    if (data_) free((void *)data_);
}
