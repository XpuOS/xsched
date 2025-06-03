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

// NEW_POLICY: A new PolicyType should be added here when creating a new policy.
enum PolicyType
{
    kPolicyTypeUnknown                     = 0,
    kPolicyTypeGlobal                      = 1,
    kPolicyTypeAppManaged                  = 2,
    kPolicyTypeInternalMax                 = 3,
       
    kPolicyTypeHighestPriorityFirst        = 4,
    kPolicyTypeUtilizationPartition        = 5,
    kPolicyTypeProcessUtilizationPartition = 6,
    kPolicyTypeEarlyDeadlineFirst          = 7,
    kPolicyTypeLaxity                      = 8,
    kPolicyTypeKEarlyDeadlineFirst         = 9,
    // NEW_POLICY: New PolicyTypes go here.

    kPolicyTypeMax,
};

class Policy
{
public:
    Policy(PolicyType type): kType(type) {}
    virtual ~Policy() = default;

    void SetSuspendFunc(OperateFunc suspend) { suspend_func_ = suspend; }
    void SetResumeFunc(OperateFunc resume) { resume_func_ = resume; }
    void SetAddTimerFunc(AddTimerFunc add_timer) { add_timer_func_ = add_timer; }

    virtual void Sched(const Status &status) = 0;
    virtual void RecvHint(std::shared_ptr<const Hint> hint) = 0;

    const PolicyType kType;

protected:
    void Suspend(XQueueHandle xqueue);
    void Resume(XQueueHandle xqueue);
    void AddTimer(const TimePoint time_point);

private:
    OperateFunc suspend_func_ = nullptr;
    OperateFunc resume_func_ = nullptr;
    AddTimerFunc add_timer_func_ = nullptr;
};

std::unique_ptr<Policy> CreatePolicy(PolicyType type);

} // namespace xsched::sched
