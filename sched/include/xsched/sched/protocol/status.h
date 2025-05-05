#pragma once

#include <mutex>
#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <condition_variable>

#include "xsched/types.h"
#include "xsched/utils/common.h"

namespace xsched::sched
{

struct XQueueStatus
{
    XQueueHandle  handle;
    XDevice       device;
    XPreemptLevel level;
    PID           pid;
    int64_t       threshold;
    int64_t       batch_size;
    bool          ready;
    bool          suspended;
    std::chrono::system_clock::time_point ready_time;
};

struct ProcessInfo
{
    PID pid;
    std::string cmdline;
};

struct ProcessStatus
{
    ProcessInfo info;
    std::unordered_set<XQueueHandle> running_xqueues;
    std::unordered_set<XQueueHandle> suspended_xqueues;
};

struct Status
{
    std::unordered_map<XQueueHandle, std::unique_ptr<XQueueStatus>> xqueue_status;
    std::unordered_map<PID, std::unique_ptr<ProcessStatus>> process_status;
};

class StatusQuery
{
public:
    StatusQuery(bool query_process) : kQueryProcess(query_process) {}
    ~StatusQuery() = default;

    void Wait();
    void Notify();
    void Reset();
    bool QueryProcess() const { return kQueryProcess; }

    std::vector<std::unique_ptr<XQueueStatus>> status_;
    std::vector<std::unique_ptr<ProcessInfo>> processes_;

private:
    const bool kQueryProcess = false; // whether to query process info at the same time
    bool ready_ = false;
    std::mutex mtx_;
    std::condition_variable cv_;
};

} // namespace xsched::sched
