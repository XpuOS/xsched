#pragma once

#include <memory>
#include <unordered_map>

#include <httplib.h>
#include <json/json.h>

#include "xsched/types.h"
#include "xsched/utils/common.h"
#include "xsched/sched/protocol/status.h"

namespace xsched::service
{

class Client
{
public:
    Client(const std::string &addr, uint16_t port);
    ~Client();

    // XQueue query
    XResult QueryXQueues(std::vector<sched::XQueueStatus> &xqueues,
                         std::unordered_map<PID, std::string> &cmdlines);

    // XQueue config
    XResult SetXQueueConfig(XQueueHandle handle, XPreemptLevel level,
                            int64_t threshold, int64_t batch_size);

    // policy
    XResult QueryPolicy(XPolicyType &policy);
    XResult SetPolicy(XPolicyType policy);

    // hints
    XResult SetPriority(XQueueHandle handle, Priority prio);
    XResult SetProcessPriority(PID pid, Priority prio);
    XResult SetUtilization(XQueueHandle handle, Utilization util);
    XResult SetProcessUtilization(PID pid, Utilization util);
    XResult SetTimeslice(Timeslice ts_us);
    XResult SetActiveWindow(PID pid, uint64_t display);

private:
    XResult SendHint(const Json::Value &request);
    XResult GetResponse(const httplib::Result &res, Json::Value &response);

    const std::string kAddr;
    const uint16_t kPort;

    Json::Reader json_reader_;
    Json::StreamWriterBuilder json_writer_;
    std::unique_ptr<httplib::Client> http_client_ = nullptr;
};

} // namespace xsched::service
