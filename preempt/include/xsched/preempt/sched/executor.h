#pragma once

#include <memory>
#include <atomic>

#include "xsched/utils/common.h"
#include "xsched/sched/protocol/operation.h"

namespace xsched::preempt
{

class SchedExecutor
{
public:
    STATIC_CLASS(SchedExecutor);

    static void Start();
    static void Stop();

    // This will be registered to scheduler by SchedAgent and be called by scheduler.
    static void Execute(std::shared_ptr<const sched::Operation> op);

private:
    static std::atomic_bool executing_;
    static void ExecuteSchedOperation(std::shared_ptr<const sched::SchedOperation> op);
    static void ExecuteConfigOperation(std::shared_ptr<const sched::ConfigOperation> op);
};

} // namespace xsched::preempt
