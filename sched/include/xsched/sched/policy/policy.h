#pragma once

#include <chrono>
#include <memory>
#include <functional>

#include "xsched/types.h"
#include "xsched/sched/protocol/hint.h"
#include "xsched/sched/protocol/status.h"

namespace xsched::sched
{

typedef std::chrono::system_clock::time_point TimePoint;
typedef std::function<void (const TimePoint)> AddTimerFunc;
typedef std::function<void (const XQueueHandle)> OperateFunc;

class Policy
{
public:
    Policy(XPolicyType type): kType(type) {}
    virtual ~Policy() = default;

    void SetSuspendFunc(OperateFunc suspend) { suspend_func_ = suspend; }
    void SetResumeFunc(OperateFunc resume) { resume_func_ = resume; }
    void SetAddTimerFunc(AddTimerFunc add_timer) { add_timer_func_ = add_timer; }

    virtual void Sched(const Status &status) = 0;
    virtual void RecvHint(std::shared_ptr<const Hint> hint) = 0;

    const XPolicyType kType;

protected:
    void Suspend(XQueueHandle xqueue);
    void Resume(XQueueHandle xqueue);
    void AddTimer(const TimePoint time_point);

private:
    OperateFunc suspend_func_ = nullptr;
    OperateFunc resume_func_ = nullptr;
    AddTimerFunc add_timer_func_ = nullptr;
};

std::unique_ptr<Policy> CreatePolicy(XPolicyType type);

} // namespace xsched::sched
