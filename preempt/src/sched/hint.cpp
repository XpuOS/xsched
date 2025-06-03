#include <memory>

#include "xsched/hint.h"
#include "xsched/utils/common.h"
#include "xsched/sched/protocol/hint.h"
#include "xsched/preempt/sched/agent.h"
#include "xsched/preempt/xqueue/xqueue.h"

using namespace xsched::sched;
using namespace xsched::preempt;

EXPORT_C_FUNC XResult XHintPriority(XQueueHandle xq, Priority prio)
{
    XResult res = XQueueManager::Exists(xq);
    if (res != kXSchedSuccess) {
        XWARN("XQueue with handle 0x%lx does not exist", xq);
        return res;
    }
    SchedAgent::SendHint(std::make_shared<PriorityHint>(xq, prio));
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult XHintUtilization(XQueueHandle xq, Utilization util)
{
    XResult res = XQueueManager::Exists(xq);
    if (res != kXSchedSuccess) {
        XWARN("XQueue with handle 0x%lx does not exist", xq);
        return res;
    }
    SchedAgent::SendHint(std::make_shared<UtilizationHint>(GetProcessId(), xq, util));
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult XHintTimeslice(Timeslice ts_us)
{
    SchedAgent::SendHint(std::make_shared<TimesliceHint>(ts_us));
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult XHintLaxity(XQueueHandle xq, Laxity lax_us, Priority lax_prio, Priority crit_prio)
{
    XResult res = XQueueManager::Exists(xq);
    if (res != kXSchedSuccess) {
        XWARN("XQueue with handle 0x%lx does not exist", xq);
        return res;
    }
    SchedAgent::SendHint(std::make_shared<LaxityHint>(xq, lax_us, lax_prio, crit_prio));
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult XHintDeadline(XQueueHandle xq, Deadline ddl_us)
{
    XResult res = XQueueManager::Exists(xq);
    if (res != kXSchedSuccess) {
        XWARN("XQueue with handle 0x%lx does not exist", xq);
        return res;
    }
    SchedAgent::SendHint(std::make_shared<DeadlineHint>(xq, ddl_us));
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult XHintKDeadline(int k)
{
    SchedAgent::SendHint(std::make_shared<KDeadlineHint>(k));
    return kXSchedSuccess;
}