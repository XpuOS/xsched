#pragma once

#include "xsched/types.h"

#ifdef __cplusplus
extern "C" {
#endif

XResult XHintPriority(XQueueHandle xq, Priority prio);
XResult XHintUtilization(XQueueHandle xq, Utilization util);
XResult XHintTimeslice(Timeslice ts_us);
XResult XHintLaxity(XQueueHandle xq, Laxity lax_us, Priority lax_prio, Priority crit_prio);
XResult XHintDeadline(XQueueHandle xq, Deadline ddl_us);
XResult XHintKDeadline(int k);

#ifdef __cplusplus
}
#endif
