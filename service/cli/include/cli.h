#pragma once

#include <memory>

#include "client.h"
#include "xsched/types.h"
#include "xsched/utils/common.h"

namespace xsched::service
{

class Cli
{
public:
    Cli(const std::string &addr, uint16_t port);
    ~Cli() = default;

    // XQueue query
    int ListXQueues();
    int TopXQueues(uint64_t interval_ms);

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
    std::unique_ptr<Client> client_ = nullptr;
};

} // namespace xsched::service
