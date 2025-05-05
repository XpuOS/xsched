#pragma once

#include "xsched/sched/scheduler/scheduler.h"

namespace xsched::sched
{

class AppManagedScheduler : public Scheduler
{
public:
    AppManagedScheduler(): Scheduler(kSchedulerAppManaged) {}
    virtual ~AppManagedScheduler() = default;

    virtual void Run() override {}
    virtual void Stop() override {}
    virtual void RecvEvent(std::shared_ptr<const Event>) override {}
};

} // namespace xsched::sched
