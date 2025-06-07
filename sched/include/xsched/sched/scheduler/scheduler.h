#pragma once

#include <memory>
#include <functional>

#include "xsched/types.h"
#include "xsched/sched/protocol/event.h"
#include "xsched/sched/protocol/operation.h"

namespace xsched::sched
{

typedef std::function<void(std::shared_ptr<const Operation>)> Executor;

class Scheduler
{
public:
    Scheduler(XSchedulerType type): kType(type) {}
    virtual ~Scheduler() = default;

    virtual void Run() = 0;
    virtual void Stop() = 0;
    virtual void RecvEvent(std::shared_ptr<const Event> event) = 0;

    XSchedulerType GetType() const { return kType; }
    void SetExecutor(Executor executor) { executor_ = executor; }

protected:
    void Execute(std::shared_ptr<const Operation> operation);

private:
    const XSchedulerType kType;
    Executor executor_ = nullptr;
};

std::shared_ptr<Scheduler> CreateScheduler(XSchedulerType scheduler, XPolicyType policy);

} // namespace xsched::sched
