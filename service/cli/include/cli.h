#pragma once

#include <memory>
#include <httplib.h>
#include <json/json.h>
#include <libipc/ipc.h>
#include <tabulate/table.hpp>

#include "xsched/types.h"
#include "xsched/sched/protocol/hint.h"
#include "xsched/sched/protocol/status.h"

namespace xsched::service
{

class Cli
{
public:
    Cli(const std::string& addr, uint16_t port);
    ~Cli();

    // XQueue query
    int ListXQueues();
    int Top(uint64_t interval_ms);

    // XQueue config
    int ConfigXQueue(XQueueHandle handle, XPreemptLevel level,
                     int64_t threshold, int64_t batch_size);

    // policy
    int QueryPolicy();
    int SetPolicy(const std::string &policy_name);

    // hints
    int SetPriority(XQueueHandle handle, Priority prio);
    int SetProcessPriority(PID pid, Priority prio);
    int SetUtilization(XQueueHandle handle, Utilization util);
    int SetProcessUtilization(PID pid, Utilization util);
    int SetTimeslice(Timeslice ts_us);

private:
    std::string ToHex(uint64_t x);
    void SendHint(const Json::Value &request);
    void GetResponse(const httplib::Result &res, Json::Value &response);

    const std::string kAddr;
    const uint16_t kPort;

    Json::Reader json_reader_;
    Json::StreamWriterBuilder json_writer_;
    std::unique_ptr<httplib::Client> client_ = nullptr;
};

} // namespace xsched::service
