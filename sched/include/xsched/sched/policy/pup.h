#pragma once

#include <set>
#include <map>
#include <list>
#include <chrono>

#include "xsched/types.h"
#include "xsched/protocol/def.h"
#include "xsched/sched/policy/policy.h"
#include "xsched/sched/protocol/hint.h"

namespace xsched::sched
{

struct RunEntry
{
    PID pid;
    bool ready;
    bool running;
    int64_t budget_us;
};

class ProcessUtilizationPartitionPolicy : public Policy
{
public:
    ProcessUtilizationPartitionPolicy(): Policy(kPolicyTypeProcessUtilizationPartition)
    { timeslice_end_ = std::chrono::system_clock::time_point::min(); }
    virtual ~ProcessUtilizationPartitionPolicy() = default;

    virtual void Sched(const Status &status) override;
    virtual void RecvHint(std::shared_ptr<const Hint> hint) override;

private:
    int64_t GetBudgetUs(PID pid);
    bool ProcessReady(PID pid, const Status &status);
    void SwitchProcess(PID pid, const Status &status);

    std::list<RunEntry> run_queue_;
    std::map<PID, Utilization> utils_;
    std::set<PID> pid_in_queue_;
    std::chrono::system_clock::time_point timeslice_end_;
    Timeslice timeslice_avg_us_ = TIMESLICE_DEFAULT;
};

} // namespace xsched::sched
