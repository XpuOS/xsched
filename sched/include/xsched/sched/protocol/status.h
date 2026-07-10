#pragma once

#include <set>
#include <map>
#include <list>
#include <mutex>
#include <memory>
#include <vector>
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
    std::set<XQueueHandle> running_xqueues;
    std::set<XQueueHandle> suspended_xqueues;
    std::set<XQueueHandle> xqueues_to_resume;
    std::set<XQueueHandle> xqueues_to_suspend;
};

struct Status
{
    std::map<XQueueHandle, std::unique_ptr<XQueueStatus>> xqueue_status;
    std::map<PID, std::unique_ptr<ProcessStatus>> process_status;
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
