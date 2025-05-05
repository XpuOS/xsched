#pragma once

#include <map>
#include <memory>
#include <thread>

#include "xsched/sched/protocol/hint.h"
#include "xsched/sched/protocol/event.h"
#include "xsched/sched/protocol/operation.h"
#include "xsched/sched/scheduler/scheduler.h"

namespace xsched::preempt
{

class SchedAgent
{
public:
    SchedAgent();
    ~SchedAgent();
    static void SendHint(std::shared_ptr<const sched::Hint> hint);
    static void SendEvent(std::shared_ptr<const sched::Event> event);

private:
    void RelayHint(std::shared_ptr<const sched::Hint> hint);
    void RelayEvent(std::shared_ptr<const sched::Event> event);

    std::shared_ptr<sched::Scheduler> scheduler_ = nullptr;
    static SchedAgent g_sched_agent;
};

} // namespace xsched::preempt
