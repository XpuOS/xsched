#pragma once

#include <memory>
#include <functional>

#include "xsched/sched/policy/policy.h"
#include "xsched/sched/protocol/event.h"
#include "xsched/sched/protocol/operation.h"

namespace xsched::sched
{

typedef std::function<void(std::shared_ptr<const Operation>)> Executor;

enum SchedulerType
{
    kSchedulerLocal      = 0,
    kSchedulerGlobal     = 1,
    kSchedulerAppManaged = 2,
};

class Scheduler
{
public:
    Scheduler(SchedulerType type): kType(type) {}
    virtual ~Scheduler() = default;

    virtual void Run() = 0;
    virtual void Stop() = 0;
    virtual void RecvEvent(std::shared_ptr<const Event> event) = 0;
    void SetExecutor(Executor executor) { executor_ = executor; }

protected:
    void Execute(std::shared_ptr<const Operation> operation);

private:
    const SchedulerType kType;
    Executor executor_ = nullptr;
};

std::shared_ptr<Scheduler> CreateScheduler();

} // namespace xsched::sched
