#include "xsched/utils/log.h"
#include "xsched/utils/str.h"
#include "xsched/utils/common.h"
#include "xsched/utils/xassert.h"
#include "xsched/preempt/sched/agent.h"
#include "xsched/preempt/sched/executor.h"

using namespace xsched::sched;

namespace xsched::preempt
{

SchedAgent SchedAgent::g_sched_agent;

SchedAgent::SchedAgent()
{
    if (scheduler_ != nullptr) return;
    SchedExecutor::Start();
    scheduler_ = CreateScheduler();
    scheduler_->SetExecutor(SchedExecutor::Execute);
    scheduler_->Run();

    std::string cmdline;
    std::ifstream cmdline_file("/proc/self/cmdline");
    if (cmdline_file.good()) {
        std::string arg;
        while (std::getline(cmdline_file, arg, '\0') && !arg.empty()) {
            cmdline = cmdline + arg + " ";
        }
    }
    cmdline_file.close();
    cmdline = ShrinkString(cmdline, ProcessCreateEvent::CmdlineCapacity() - 1);
    auto event = std::make_shared<ProcessCreateEvent>(cmdline);
    scheduler_->RecvEvent(event);
}

SchedAgent::~SchedAgent()
{
    if (scheduler_ == nullptr) return;
    SchedExecutor::Stop();
    auto event = std::make_shared<ProcessDestroyEvent>();
    scheduler_->RecvEvent(event);
    scheduler_->Stop();
    scheduler_ = nullptr;
}

void SchedAgent::SendHint(std::shared_ptr<const sched::Hint> hint)
{
    g_sched_agent.RelayHint(hint);
}

void SchedAgent::SendEvent(std::shared_ptr<const sched::Event> event)
{
    g_sched_agent.RelayEvent(event);
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

} // namespace xsched::preempt
