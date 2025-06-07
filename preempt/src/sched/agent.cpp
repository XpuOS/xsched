#include "xsched/types.h"
#include "xsched/utils/log.h"
#include "xsched/utils/str.h"
#include "xsched/utils/xassert.h"
#include "xsched/protocol/def.h"
#include "xsched/protocol/names.h"
#include "xsched/preempt/sched/agent.h"
#include "xsched/preempt/sched/executor.h"
#include "xsched/sched/scheduler/local.h"

using namespace xsched::sched;
using namespace xsched::protocol;

namespace xsched::preempt
{

SchedAgent SchedAgent::g_sched_agent;

SchedAgent::SchedAgent()
{
    if (scheduler_ != nullptr) return;

    XSchedulerType scheduler_type = kSchedulerUnknown;
    XPolicyType policy_type = kPolicyUnknown;
    char *scheduler_str = std::getenv(XSCHED_SCHEDULER_ENV_NAME);
    char *policy_str = std::getenv(XSCHED_POLICY_ENV_NAME);
    if (scheduler_str) scheduler_type = GetSchedulerType(scheduler_str);
    if (policy_str) policy_type = GetPolicyType(policy_str);

    SchedExecutor::Start();
    InitScheduler(scheduler_type, policy_type);
}

SchedAgent::~SchedAgent()
{
    if (scheduler_ == nullptr) return;
    SchedExecutor::Stop();
    FiniScheduler();
}

void SchedAgent::RelayHint(std::shared_ptr<const sched::Hint> hint)
{
    if (scheduler_ == nullptr) {
        XWARN("scheduler not initialized, hint type(%d) dropped", hint->Type());
        return;
    }
    scheduler_->RecvEvent(std::make_shared<HintEvent>(hint));
}

void SchedAgent::RelayEvent(std::shared_ptr<const Event> event)
{
    if (scheduler_ == nullptr) {
        XWARN("scheduler not initialized, event type(%d) dropped", event->Type());
        return;
    }
    scheduler_->RecvEvent(event);
}

void SchedAgent::InitScheduler(XSchedulerType scheduler_type, XPolicyType policy_type)
{
    if (scheduler_ != nullptr) return;
    
    auto tmp_scheduler = CreateScheduler(scheduler_type, policy_type);
    tmp_scheduler->SetExecutor(SchedExecutor::Execute);
    tmp_scheduler->Run();

    static const std::string kCmdline = []() -> std::string {
        std::string cmdline;
        std::ifstream cmdline_file("/proc/self/cmdline");
        if (cmdline_file.good()) {
            std::string arg;
            while (std::getline(cmdline_file, arg, '\0') && !arg.empty()) {
                cmdline = cmdline + arg + " ";
            }
        }
        cmdline_file.close();
        return ShrinkString(cmdline, ProcessCreateEvent::CmdlineCapacity() - 1);
    }();

    auto event = std::make_shared<ProcessCreateEvent>(kCmdline);
    tmp_scheduler->RecvEvent(event);

    scheduler_ = tmp_scheduler;
}

void SchedAgent::FiniScheduler()
{
    if (scheduler_ == nullptr) return;
    auto tmp_scheduler = scheduler_;
    scheduler_ = nullptr;

    auto event = std::make_shared<ProcessDestroyEvent>();
    tmp_scheduler->RecvEvent(event);
    tmp_scheduler->Stop();
}

XResult SchedAgent::ReplaceScheduler(XSchedulerType scheduler_type, XPolicyType policy_type)
{
    if (scheduler_type <= kSchedulerUnknown || scheduler_type >= kSchedulerMax) {
        XWARN("invalid scheduler type: %d", scheduler_type);
        return kXSchedErrorInvalidValue;
    }

    if (scheduler_ == nullptr) {
        XWARN("scheduler not initialized, replace scheduler failed");
        return kXSchedErrorUnknown;
    }

    if (scheduler_->GetType() == scheduler_type) {
        if (scheduler_type != kSchedulerLocal) return kXSchedSuccess; // no need to replace
        if (policy_type <= kPolicyUnknown || policy_type >= kPolicyMax) {
            XWARN("invalid policy type: %d", policy_type);
            return kXSchedErrorInvalidValue;
        }
        // only replace the policy
        auto lcl_scheduler = std::dynamic_pointer_cast<LocalScheduler>(scheduler_);
        XASSERT(lcl_scheduler != nullptr, "scheduler is not a local scheduler");
        lcl_scheduler->SetPolicy(policy_type);
        return kXSchedSuccess;
    }

    // should replace the scheduler
    FiniScheduler();
    InitScheduler(scheduler_type, policy_type);
    return kXSchedSuccess;
}

} // namespace xsched::preempt
